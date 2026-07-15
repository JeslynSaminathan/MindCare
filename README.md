# MindCare

An explainable AI mental health companion built for a Final Year Project. MindCare
combines a fine-tuned DistilBERT intent classifier, a LIME-based explainability
layer, a deterministic rule-based crisis detector, and Qwen 2.5 for empathetic
response generation, wrapped in a Flask web app with an anonymous session model.

> **MindCare is not a diagnostic or clinical tool.** It never diagnoses a mental
> health condition and always defers to professional care and crisis services
> where appropriate. See [Safety design](#safety-design) below.

---

## 1. Architecture overview

```
User message
     │
     ▼
crisis_detector.py  ── deterministic, rule-based, tiered pattern matching
     │  (if crisis detected: return FIXED template response, stop here)
     ▼
chatbot.py: DistilBERT intent classifier ── predicts intent + confidence
     │  (defense-in-depth: if classifier alone lands on "crisis" with
     │   reasonable confidence, also stop here with a fixed resource-
     │   offering response, instead of generating freely)
     ▼
chatbot_explainer.py: LIME ── explains classifier decision (token weights)
     │
     ▼
chatbot.py: Qwen 2.5 generator ── synthesizes empathetic reply using intent
     │        context notes + system-prompt safety guardrails + history
     ▼
conversation_logger.py ── logs turn to SQLite (anonymous session_id only)
     │
     ▼
Flask JSON response ─→ chat.js renders reply + updates XAI panel
```

**Two independent safety layers, not one:**

1. **Rule-based gate (`crisis_detector.py`), runs first.** Every pattern that
   can trigger a crisis-tier response is a plain-text regex you can read in
   the file — there is no opaque model weight deciding whether someone in
   danger gets redirected to a hotline. When it fires, the response text is a
   fixed, reviewed template, never LLM-generated output.
2. **Classifier-based fallback (`chatbot.py`, step 2a), runs second.** If the
   rule-based gate doesn't fire but the DistilBERT classifier independently
   predicts the `crisis` intent with reasonable confidence, the turn is still
   routed to a fixed, cautious resource-offering response rather than free
   generation. This catches phrasings the regex patterns don't cover yet.

Only messages that clear *both* layers ever reach the Qwen 2.5 generator.

## 2. Directory structure

```
MindCare/
├── app.py                    # Flask app & API routes
├── train_distilbert.py       # Fine-tunes distilbert-base-uncased on data/intents.json
├── chatbot.py                # Orchestrates crisis check -> classifier -> LIME -> Qwen 2.5
├── chatbot_explainer.py      # LIME wrapper around the intent classifier
├── crisis_detector.py        # Tiered, rule-based crisis detection & response templates
├── preprocessing.py          # Shared text cleaning / tokenization helpers
├── conversation_logger.py    # SQLite adapter (anonymous session + message logging)
├── requirements.txt
├── README.md
├── data/
│   ├── intents.json          # Intent patterns, context notes, canned fallback responses
│   ├── label2id.json
│   └── id2label.json
├── templates/
│   ├── welcome.html          # Anonymous session gate
│   └── index.html            # Chat UI + live XAI panel
└── static/
    ├── css/style.css
    ├── js/welcome.js
    ├── js/chat.js
    └── images/logo.png
```

## 3. Intent taxonomy

`data/intents.json` defines **24 intents** the DistilBERT classifier is trained
against. This taxonomy was curated from a combination of hand-written seed
data and a public Kaggle mental-health conversational dataset, with meta/
engine-specific artifacts of the source dataset (e.g. `no-response`, `wrong`,
`default`, 32 near-duplicate single-pattern "fact-N" tags) either dropped or
consolidated into broader categories:

`about_bot`, `anxiety`, `bullying`, `coping_strategies`, `crisis`,
`frustration_with_bot`, `general_support`, `goodbye`, `greeting`, `grief`,
`help_request`, `loneliness`, `meditation_request`, `mental_health_facts`,
`mental_health_info`, `positive_mood`, `relationship_issues`,
`reluctant_to_share`, `sadness`, `self_esteem`, `sleep_issues`, `small_talk`,
`stress`, `thanks`.

**Important curation note on `crisis`:** the source dataset's own canned
responses for suicide-related phrases were not safe to reuse verbatim (one
example cheerfully replied "the time has come for you to show the world...!!"
to a suicidal-ideation phrase). Those responses were discarded entirely.
Instead, the *phrasings* were mined into `crisis_detector.py`'s regex pattern
matrix, and `crisis` in `data/intents.json` carries only MindCare's own
reviewed, cautious response text — used solely as the step-2a defense-in-depth
fallback described above, never as free-generation material.

## 4. Setup

### 4.1 Requirements

- Python 3.11+
- ~4 GB disk for DistilBERT + Qwen2.5-1.5B-Instruct weights (more if you switch
  to the 7B variant)
- A CUDA GPU is optional but strongly recommended for the generator; CPU works
  for the classifier and for the 1.5B generator at reduced speed.

### 4.2 Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4.3 Train the intent classifier

```bash
python train_distilbert.py --epochs 8 --batch-size 16
```

This fine-tunes `distilbert-base-uncased` on the patterns in
`data/intents.json` (with light built-in augmentation) and saves the best
checkpoint to `models/distilbert-intent/`. `app.py` will fall back to an
untrained base model with a warning if this step is skipped, so the app is
still runnable end-to-end for a demo, but intent accuracy will be poor until
you train.

### 4.4 Run the app

```bash
export FLASK_DEBUG=1                 # optional, for local dev
python app.py
```

Visit `http://localhost:5000/`. The first load downloads the Qwen 2.5 weights
if they aren't cached yet (this can take a while on first run).

To skip loading the generator entirely (fast boot, canned responses only —
useful for frontend/API development):

```bash
export MINDCARE_SKIP_GENERATOR=1
python app.py
```

To use the larger `Qwen/Qwen2.5-7B-Instruct` model instead of the default
1.5B:

```bash
export MINDCARE_GENERATOR_MODEL="Qwen/Qwen2.5-7B-Instruct"
python app.py
```

## 5. API reference

| Method | Route                     | Description                                   |
|--------|---------------------------|------------------------------------------------|
| GET    | `/`                       | Welcome / anonymous session gate               |
| GET    | `/chat?sid=<id>`          | Chat UI (redirects to `/` if session invalid)  |
| POST   | `/api/session/start`      | Create anonymous session. Body: `{age_range?, gender?}` |
| POST   | `/api/chat`                | Send a message. Body: `{session_id, message}`  |
| GET    | `/api/history/<sid>`      | Retrieve prior turns for a session             |
| GET    | `/api/health`             | Liveness probe                                 |

Example `/api/chat` response:

```json
{
  "reply": "That sounds like a lot to carry. Can you tell me more about what's weighing on you most?",
  "intent": "stress",
  "confidence": 0.8734,
  "crisis_tier": "none",
  "explanation": {
    "predicted_intent": "stress",
    "confidence": 0.8734,
    "weights": [
      {"token": "overwhelmed", "weight": 0.31},
      {"token": "pressure", "weight": 0.22}
    ]
  }
}
```

## 6. Safety design

- **No diagnosis, ever.** The Qwen 2.5 system prompt (in `chatbot.py`)
  explicitly forbids diagnostic language and medical/medication advice, and
  every screen in the UI carries a visible disclaimer.
- **Two independent crisis layers.** See Section 1. The rule-based gate runs
  first and is fully auditable; the classifier-based fallback catches
  phrasings the regexes miss. Neither layer's response text is ever
  LLM-generated.
- **Anonymous by design.** Sessions are identified only by a random UUID.
  Age range and gender are optional and stored only if the user opts in.
  No names, emails, or device fingerprints are collected.
- **Auditability.** Every assistant turn is logged with its predicted intent,
  confidence, and crisis tier, so the project team can review both model
  behavior and the safety gates' trigger history.

## 7. Limitations & FYP scope notes

- The intent taxonomy and training data in `data/intents.json` are curated
  at demonstration scale (24 tags, tens to low-hundreds of patterns per tag);
  this is appropriate for a Final Year Project baseline, not a
  production-scale corpus.
- The crisis pattern matrix uses substring/regex matching, which will miss
  crisis language it hasn't been written to catch, and can occasionally
  false-positive on figurative speech (one such false positive, an overly
  broad "my time has come" pattern, was found and removed during curation —
  see `crisis_detector.py` for the current pattern set). It is designed to
  fail toward caution (over-triggering) rather than silence.
- This system has not been clinically validated and must not be deployed as
  a substitute for professional mental health services.
