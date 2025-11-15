
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class DeBERTaClient:
    """Client for zero-shot classification using DeBERTa-v3-base-mnli."""

    def __init__(
        self,
        model: str = "MoritzLaurer/DeBERTa-v3-base-mnli",
        device: str | None = None,
    ):
        self.model_name = model
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.model.to(self.device)
        self.label_names = ["entailment", "neutral", "contradiction"]

    def classify(
        self,
        premise: str,
        hypothesis: str,
        return_probabilities: bool = True,
    ) -> dict[str, float]:
        """Classify the relationship between premise and hypothesis."""
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0]
        probabilities = torch.softmax(logits, -1).tolist()

        if return_probabilities:
            return {
                name: round(float(prob) * 100, 1)
                for prob, name in zip(probabilities, self.label_names)
            }
        else:
            return {
                name: float(prob) for prob, name in zip(probabilities, self.label_names)
            }

    def classify_batch(
        self,
        premises: list[str],
        hypotheses: list[str],
        return_probabilities: bool = True,
    ) -> list[dict[str, float]]:
        """Classify multiple premise-hypothesis pairs."""
        if len(premises) != len(hypotheses):
            raise ValueError("Number of premises must match number of hypotheses")

        results = []
        for premise, hypothesis in zip(premises, hypotheses):
            result = self.classify(premise, hypothesis, return_probabilities)
            results.append(result)

        return results

    def get_prediction(self, premise: str, hypothesis: str) -> str:
        """Get the most likely label for the premise-hypothesis pair."""
        scores = self.classify(premise, hypothesis, return_probabilities=False)
        return max(scores, key=scores.get)
