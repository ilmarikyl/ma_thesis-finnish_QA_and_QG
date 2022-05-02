"""
Script to prepare the text dataset of triplet and a triplet is of:
    - context
    - question
    - answer

No NLP in the procedures of this script.

Prerequisite
--------------
Two files from SQuAD v1.1:
    squad/train-v1.1.json
    squad/dev-v1.1.json

Outputs
--------------
    para_60k_dev.json
    para_60k_test.json
    para_60k_train.json
    patched_train-v1.1.json
    patched_dev-v1.1.json

TODO
----------------
Fix this script

"""
from utils.preprocessing.text_dataset import SQuADV1TextDataset
from utils.preprocessing.squad import ParagraphSQuAD60kSplitsBuilder
from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass
class Configs:
    data_dir: str = "../../datasets"
    output_dir: str = "./prep_txt_ds"	
    split_info_dir: str = "../../datasets/qg_split_info"


if __name__ == "__main__":
    cfg = Configs()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Work directory: {output_dir}")
    # for s in ["train", "dev"]:
        # ds = SQuADV1TextDataset(cfg.data_dir, split=s)
        # # patched_ds = PatchedSQuADTextDataset(ds)
        # # -------------------------------------------------
        # filepath = output_dir / f"{s}-fin-v1.1.json"
        # ds.to_json(filepath)
        # logging.info(f"Wrote data (not patched because Finnish) into {filepath}")
        # b = ParagraphSQuAD60kSplitsBuilder(cfg.split_info_dir, output_dir, ds)
        # b.save_examples()

    ds = SQuADV1TextDataset(cfg.data_dir)
    filepath = output_dir / f"ALL-fin-v1.1.json"
    ds.to_json(filepath)
    logging.info(f"Wrote data (not patched because Finnish) into {filepath}")
    b = ParagraphSQuAD60kSplitsBuilder(cfg.split_info_dir, output_dir, ds)
    b.save_examples()
