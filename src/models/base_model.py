from abc import ABC, abstractmethod
from typing import Dict, List
class BaseManipulationDetector(ABC):

    @abstractmethod
    def predict(self, context: str, current_utterance: str) -> Dict:
        """
        Input:
            context: str (previous turns)
            current_utterance: str

        Output:
            {
                "manipulation": float (probability),
                "techniques": {label: probability}
            }
        """
        pass

    def batch_predict(self, inputs: List[str]):
        pass

    def evaluate(self, dataset, metrics_fn):
        pass