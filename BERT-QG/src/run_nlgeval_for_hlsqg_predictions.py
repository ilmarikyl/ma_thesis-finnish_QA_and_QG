"""
                                Script Documentation

This script is to generate questions specific for the model `hlsqg`


This script evaluate the following experiments:

1. HLSQG + sequential decoding
"""
import json, utils, argparse, time, dataclasses
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from nlgeval import NLGEval
from utils.logging import logging
from utils.logging import set_logfile_output
from utils.hf_argparser import HfArgumentParser
from utils.data.datasets.wrapper import CausalCondTextGenEvalDatasetWrapper
from utils.eval import clean_hypothesis
from utils.eval.factory import BaseEvaluationFactory
from utils.eval.arguments import BaseEvalScriptArguments
from utils.eval.decoding import CausalMLMBSCondTokenDecoder

LOG_FILE_NAME = "FinGTP-QG-v2_cp_64608.log"


@dataclass
class ScriptArguments(BaseEvalScriptArguments):
    pass


class EvaluationFactory(BaseEvaluationFactory):

    def __init__(self, configs: ScriptArguments):
        super().__init__(configs)

    def create_dataset(self):
        """
        return a simple left-to-right generation dataset
        """
        ret = CausalCondTextGenEvalDatasetWrapper(self.text_dataset, self.tokenizer)
        return ret

    def create_decoder(self):
        ret = CausalMLMBSCondTokenDecoder(
            self.model, self.tokenizer,
            no_repeat_ngram_size=2,
            decode_length_known=False,
            num_return_sequences=1)
        return ret

    def create_task_name(self) -> str:
        ret = "hlsqg"
        return ret


if __name__ == "__main__":

    # ---------------------------------------
    # Pick up the arguments
    # ---------------------------------------
    parser = HfArgumentParser((ScriptArguments,))
    configs, = parser.parse_args_into_dataclasses()

    # ---------------------------------------
    # logger
    # ---------------------------------------
    logger = logging.getLogger()
    logging_dir = Path(configs.logging_dir)
    logging_dir.mkdir(parents=True, exist_ok=True)
    logfile = logging_dir / LOG_FILE_NAME
    set_logfile_output(logfile)

    with open('ginGPT_QG-v2_checkpoint-64608_preds.json', 'r', encoding='utf8') as pred_f:
        results_dict = json.load(pred_f)

    results = results_dict["data"]
    
    # --------------------------------------------------
    # run nlg_eval to get n-gram overlapping metrics
    n = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True)

    # build refs vs hyps
    ref_list = []
    hyp_list = []

    for rec in results:
        ref_list.append(rec["reference"])
        hyp_list.append(rec["hypothesis"])

    stime = time.time()
    metrics = n.compute_metrics(ref_list=[ref_list], hyp_list=hyp_list)
    etime = time.time()
    eval_duration = etime - stime

    output = {
        "task_name": 'hlsqg',
        "evalution_configs": dataclasses.asdict(configs),
        "nlg-eval_duration": eval_duration,
        "batch_size": configs.batch_size,
        "metrics": metrics,
        "results": results,
    }

    output_fname = "FinGTP-QG-v2_cp_64608.json"
    output_path = logging_dir / output_fname
    logging.info(f"Saving the results to {output_path}...")
    utils.save_as_json(output_path, output)
