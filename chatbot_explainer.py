"""
chatbot_explainer.py

Wraps lime.lime_text.LimeTextExplainer around the DistilBERT intent
classifier so the frontend can show *why* a message was routed to a given
intent -- per-token contribution weights that drive the live XAI panel in
index.html.

This module never touches crisis detection: crisis routing in
crisis_detector.py happens deterministically upstream and is never explained
via LIME, since its output must remain a fixed, auditable template rather
than something a local surrogate model reasons about.
"""

from typing import List, Dict, Tuple

import numpy as np
from lime.lime_text import LimeTextExplainer

from preprocessing import clean_text


class IntentExplainer:
    """Generates LIME-based explanations for a single classifier prediction."""

    def __init__(self, model, tokenizer, id2label: Dict[int, str], device, max_length: int = 64):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.device = device
        self.max_length = max_length
        self.class_names = [id2label[i] for i in sorted(id2label.keys())]
        self.explainer = LimeTextExplainer(class_names=self.class_names)

    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """LIME calls this repeatedly with perturbed versions of the input
        text; it must return an (n_samples, n_classes) probability matrix.
        """
        import torch  # local import keeps torch optional at module import time

        cleaned = [clean_text(t) for t in texts]
        encodings = self.tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(**encodings).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def explain(self, text: str, num_features: int = 8, num_samples: int = 500) -> Dict:
        """Return a JSON-serializable explanation payload.

        {
          "predicted_intent": str,
          "confidence": float,
          "weights": [{"token": str, "weight": float}, ...]   # sorted desc by |weight|
        }
        """
        cleaned = clean_text(text)
        probs = self._predict_proba([cleaned])[0]
        predicted_idx = int(np.argmax(probs))
        predicted_label = self.id2label[predicted_idx]
        confidence = float(probs[predicted_idx])

        explanation = self.explainer.explain_instance(
            cleaned,
            self._predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=[predicted_idx],
        )

        weights: List[Tuple[str, float]] = explanation.as_list(label=predicted_idx)
        weight_payload = [
            {"token": token, "weight": round(float(weight), 4)}
            for token, weight in sorted(weights, key=lambda x: abs(x[1]), reverse=True)
        ]

        return {
            "predicted_intent": predicted_label,
            "confidence": round(confidence, 4),
            "weights": weight_payload,
        }
