
from .text_dataset import SQuADTextDataset
from .text_dataset import SquadExample
from typing import List
import logging
import json

class PatchedSQuADTextDataset(SQuADTextDataset):

    def __init__(self, ds: SQuADTextDataset, blocklist_file: str, patched_questions: str):
        # rename myself
        self.__class__.__name__ = f"Patched{ds.__class__.__name__}"
        self.logger = logging.getLogger(self.__class__.__name__)
        self.examples: List[SquadExample] = list(ds.examples)

    @staticmethod
    def load_json(file):
        with open(file, "r", encoding="utf-8") as f:
            ret = json.load(f)
        return ret
