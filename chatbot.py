"""
chatbot.py

Core logic orchestrator for MindCare.

Pipeline for every incoming user message:

    1. crisis_detector.assess_message()  -- deterministic, rule-based gate.
       If crisis language is detected, a FIXED template response is returned
       immediately and NEITHER the classifier NOR the generator is invoked
       on that turn. This is a hard short-circuit, not a soft preference.
    2. DistilBERT intent classifier -- predicts an intent label + confidence
       over the taxonomy in data/intents.json.
       2a. Defense in depth: if the classifier alone predicts the "crisis"
           label (even though step 1 did not flag the message), the turn is
           routed to a fixed, cautious resource-offering response instead of
           free generation. This catches phrasings the rule-based gate's
           regex patterns don't cover yet.
    3. chatbot_explainer.IntentExplainer -- (optional per-call) produces a
       LIME explanation of the classifier's decision for the XAI panel.
    4. Qwen 2.5 generator -- takes the predicted intent's context_notes +
       recent conversation history and synthesizes an empathetic, natural
       reply. The system prompt hard-codes the safety guardrails (no
       diagnosis, no clinical claims, always defer to professional care).

This module never lets the generator see or respond to messages that either
safety layer (1 or 2a) has flagged.
"""

import json
import os
from typing import Dict, List, Optional, Any

import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
)

import crisis_detector
from preprocessing import clean_text, truncate_for_model
from chatbot_explainer import IntentExplainer

DATA_DIR = "data"
CLASSIFIER_DIR = os.path.join("models", "distilbert-intent")

# Use the 1.5B instruct variant by default for resource-constrained
# environments; swap to Qwen/Qwen2.5-7B-Instruct if you have the VRAM.
GENERATOR_MODEL_NAME = os.environ.get("MINDCARE_GENERATOR_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

LOW_CONFIDENCE_THRESHOLD = 0.35
CRISIS_INTENT_TAG = "crisis"

SYSTEM_PROMPT = """You are MindCare, a warm, empathetic supportive-listening companion.

Hard rules you must never break:
- You are NOT a therapist, doctor, or clinician. NEVER diagnose a mental health condition \
or use diagnostic language (e.g. "you have depression", "this sounds like an anxiety disorder").
- NEVER claim to replace professional care. If appropriate, gently suggest that a licensed \
professional could help, without being pushy about it every single turn.
- NEVER give medical, medication, or dosage advice.
- Keep responses conversational, warm, and concise (2-5 sentences). Ask at most one \
follow-up question per turn.
- Validate the user's feelings without being saccharine or dismissive.
- You are told the user's likely conversational intent and short contextual notes for it. \
Use these as guidance for tone and direction, not as a script to recite verbatim.
"""


class MindCareChatbot:
    def __init__(self, load_generator: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(os.path.join(DATA_DIR, "id2label.json")) as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}
        with open(os.path.join(DATA_DIR, "intents.json")) as f:
            self.intents_data = {i["tag"]: i for i in json.load(f)["intents"]}

        self._load_classifier()

        self.explainer = IntentExplainer(
            model=self.classifier_model,
            tokenizer=self.classifier_tokenizer,
            id2label=self.id2label,
            device=self.device,
        )

        self.generator_tokenizer = None
        self.generator_model = None
        if load_generator:
            self._load_generator()

    # -- Model loading --------------------------------------------------------

    def _load_classifier(self) -> None:
        if os.path.isdir(CLASSIFIER_DIR):
            model_path = CLASSIFIER_DIR
        else:
            # Falls back to the base (untrained-head) model so the app is
            # still runnable end-to-end before train_distilbert.py has been
            # run; classifier accuracy will be poor until fine-tuned.
            print(
                f"[chatbot] WARNING: {CLASSIFIER_DIR} not found. "
                "Run train_distilbert.py first for accurate intent classification. "
                "Falling back to base distilbert-base-uncased with a random head."
            )
            model_path = "distilbert-base-uncased"

        self.classifier_tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.classifier_model = DistilBertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(self.id2label),
        ).to(self.device)
        self.classifier_model.eval()

    def _load_generator(self) -> None:
        print(f"[chatbot] Loading generator model: {GENERATOR_MODEL_NAME}")
        self.generator_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
        self.generator_model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_MODEL_NAME,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.generator_model.eval()

    # -- Intent classification -------------------------------------------------

    @torch.no_grad()
    def classify_intent(self, text: str) -> Dict[str, Any]:
        cleaned = clean_text(text)
        encoding = self.classifier_tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(self.device)

        logits = self.classifier_model(**encoding).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(torch.argmax(probs).item())

        return {
            "intent": self.id2label[pred_idx],
            "confidence": float(probs[pred_idx].item()),
        }

    # -- Generation --------------------------------------------------------

    def _build_messages(
        self, user_text: str, intent_tag: str, history: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        intent_info = self.intents_data.get(intent_tag, {})
        context_notes = intent_info.get("context_notes", "")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if history:
            for turn in history[-6:]:  # last 3 exchanges for context
                role = "user" if turn["role"] == "user" else "assistant"
                messages.append({"role": role, "content": turn["content"]})

        intent_guidance = (
            f"[Detected user intent: {intent_tag}. Context notes for this intent: "
            f"{context_notes}]\n\nUser message: {user_text}"
        )
        messages.append({"role": "user", "content": intent_guidance})
        return messages

    def generate_reply(
        self, user_text: str, intent_tag: str, history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        if self.generator_model is None:
            # Generator not loaded (e.g. lightweight/CI mode) -- fall back to
            # a canned response from the intent's own response bank so the
            # pipeline remains functional.
            responses = self.intents_data.get(intent_tag, {}).get("responses", [])
            return responses[0] if responses else (
                "I'm here and listening. Can you tell me a bit more about what's going on?"
            )

        messages = self._build_messages(user_text, intent_tag, history)
        prompt = self.generator_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.generator_tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.generator_model.generate(
                **inputs,
                max_new_tokens=180,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.15,
                pad_token_id=self.generator_tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        reply = self.generator_tokenizer.decode(generated, skip_special_tokens=True).strip()
        return reply if reply else "I'm here, and I'm listening. Can you tell me more?"

    # -- Top-level entry point --------------------------------------------------

    def handle_message(
        self,
        user_text: str,
        history: Optional[List[Dict[str, str]]] = None,
        include_explanation: bool = True,
    ) -> Dict[str, Any]:
        """Full pipeline for one user turn. Returns a dict consumed directly
        by the /api/chat Flask route.
        """
        user_text = truncate_for_model(user_text)

        # --- Step 1: hard crisis gate (rule-based, deterministic) ----------
        assessment = crisis_detector.assess_message(user_text)
        if assessment.is_crisis:
            return {
                "reply": crisis_detector.build_crisis_response(assessment),
                "intent": "crisis",
                "confidence": None,
                "crisis_tier": assessment.tier.value,
                "explanation": None,
            }

        # --- Step 2: intent classification --------------------------------
        classification = self.classify_intent(user_text)
        intent_tag = classification["intent"]
        confidence = classification["confidence"]

        # --- Step 2a: defense-in-depth classifier-only crisis fallback -----
        # The rule-based gate above did not fire, but if the trained
        # classifier itself independently lands on "crisis" with reasonable
        # confidence, don't hand this to free generation. Offer resources
        # instead of guessing. This is deliberately a softer response than
        # the WATCH tier template, since the more trustworthy rule-based
        # signal did not corroborate it.
        if intent_tag == CRISIS_INTENT_TAG and confidence >= LOW_CONFIDENCE_THRESHOLD:
            explanation = None
            if include_explanation:
                try:
                    explanation = self.explainer.explain(user_text)
                except Exception as exc:
                    print(f"[chatbot] LIME explanation failed: {exc}")
            return {
                "reply": crisis_detector.CLASSIFIER_ONLY_CRISIS_RESPONSE,
                "intent": "crisis",
                "confidence": round(confidence, 4),
                "crisis_tier": "watch",
                "explanation": explanation,
            }

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            intent_tag = "general_support"

        # --- Step 3: optional XAI explanation -------------------------------
        explanation = None
        if include_explanation:
            try:
                explanation = self.explainer.explain(user_text)
            except Exception as exc:  # LIME failures should never break chat
                print(f"[chatbot] LIME explanation failed: {exc}")
                explanation = None

        # --- Step 4: generation --------------------------------------------
        reply = self.generate_reply(user_text, intent_tag, history)

        return {
            "reply": reply,
            "intent": intent_tag,
            "confidence": round(confidence, 4),
            "crisis_tier": "none",
            "explanation": explanation,
        }
