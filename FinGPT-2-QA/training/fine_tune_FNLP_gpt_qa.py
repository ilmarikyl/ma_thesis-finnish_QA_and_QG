from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline, Trainer, TrainingArguments
from torch.utils.data import Dataset
from collections import Counter
from collections import defaultdict
import torch, re, json, argparse, sys


class SQuADDataset(Dataset):
    def __init__(self, passage_list, question_list, answer_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        # map_label = 

        for passage, question, answer in zip(passage_list, question_list, answer_list):
            answer = answer['text']

            prep_txt = f'<bos>Konteksti: {passage}\nKysymys: {question}\nVastaus: {answer}<eos>'

            encodings_dict = tokenizer(prep_txt, truncation=True, max_length=max_length, padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
            self.labels.append(answer) # !!

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]

def select_answer(answers):
        '''
        We select answers using the following rules:
        1. voting
        2. the shortest one.
        '''
        if len(answers) == 1:
            return answers[0]

        # Vote for the popular answer
        start_pos: dict = defaultdict(list)
        votes: Counter = Counter()
        for ans_dict in answers:
            answer_text = ans_dict["text"]
            ans_char_start_pos = ans_dict["answer_start"]
            start_pos[answer_text].append(ans_char_start_pos)
            votes[answer_text] += 1

        # if we have agreement (i.e. # of votes != 1)
        ans, n_vote = votes.most_common(1)[0]
        if n_vote != 1:
            return {
                "text": ans,
                "answer_start": start_pos[ans][0]
            }

        # if equal votes, select the shortest one
        min_len = 9999
        idx = -1
        for i, ans_dict in enumerate(answers):
            len_ = len(ans_dict["text"])
            if len_ > min_len:
                idx = i
                min_len = len_
        ret = {
            "text": answers[idx]["text"],
            "answer_start": answers[idx]["answer_start"]
        }
        return ret

    
def load_squad_dataset(path, tokenizer):

    with open(path, "rb") as f:
        squad_dict = json.load(f)

    contexts, questions, answers = [], [], []

    for group in squad_dict["data"]:
        for passage in group["paragraphs"]:
            context = passage["context"]

            for qa in passage["qas"]:

                if qa["is_impossible"]:
                    continue
                
                question = qa["question"]
                answer = select_answer(qa["answers"])

                contexts.append(context)
                questions.append(question)
                answers.append(answer)

    train_dataset = SQuADDataset(contexts, questions, answers, tokenizer, max_length=1024)

    return train_dataset


def main(args):

    model_name = "Finnish-NLP/gpt2-medium-finnish"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    torch.manual_seed(42)

    # train_dataset = load_squad_dataset("../../../datasets/qg_train_split-64604.json", tokenizer)
    # eval_dataset = load_squad_dataset("../../../datasets/qg_dev_split-4902.json", tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print('TOKENIZER:', tokenizer)
    print('---')
    print('MODEL:', model)

    # setup GPU/CPU
    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # move model over to detected device
    model.to(device)


    training_args = TrainingArguments(output_dir=f'/scratch/project_2001403/ilmariky/{args.exp}',
                                    num_train_epochs=args.epochs, logging_steps=100, overwrite_output_dir=True,
                                    save_strategy="epoch", evaluation_strategy="steps", eval_steps=4000,
                                    per_device_train_batch_size=2, per_device_eval_batch_size=2, warmup_steps=100,
                                    weight_decay=0.01, logging_dir=f'/scratch/project_2001403/ilmariky/{args.exp}/logs')

    print('training_args:', training_args)
    exit()


    Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
            data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                        'attention_mask': torch.stack([f[1] for f in data]),
                                        'labels': torch.stack([f[0] for f in data])}).train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Placeholder')
    parser.add_argument("--exp", required=True,
                        help="Name for experiment. Used in saving model as the output dir name.")
    parser.add_argument("--epochs", required=True, type=int, choices=range(1,21), metavar="[1-20]",
                        help="How many epoch the model will be trained. Choose between 1 and 20")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()

    main(args)
                            