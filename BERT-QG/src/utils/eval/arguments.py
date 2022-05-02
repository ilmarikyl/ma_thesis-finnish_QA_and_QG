from dataclasses import dataclass, field

@dataclass
class BaseEvalScriptArguments:
    """
    This evaluation script basic argument
    """

    logging_dir: str = field(
        metadata={
            "help": "The logging folder for evaluation."
        }
    )

    model_path: str = field(
        default="FinBERT-hlsqg-p60k-causal-v2-out",
        metadata={
            "help": "The folder of the trained `BertForMaskedLM` model"
        }
    )

    batch_size: int = field(
        default=1,
        metadata={
            "help": "Batch size of the input sequences to feed into the model"
        }
    )

    txt_ds_path: str = field(
        default="cached_txtds/test_ds",
        metadata={
            "help": "The text dataset in numpy data format"
        }
    )

    tokenizer_name: str = field(
        # default="bert-base-uncased",
        # default="TurkuNLP/bert-base-finnish-cased-v1",
        default="bert-base-multilingual-cased",
        metadata={
            "help": (
                "The name of the pretrained tokenizer. "
                "Will download the tokenizer data from huggingface storage."
            )
        }
    )
