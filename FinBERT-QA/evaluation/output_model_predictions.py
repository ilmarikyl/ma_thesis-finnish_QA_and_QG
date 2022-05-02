import json, torch, argparse, sys

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }

    def __len__(self):
        return len(self.encodings.input_ids)


def read_dataset(path, incl_impossibles):
    with open(path, "rb") as f:
        squad_dict = json.load(f)

    contexts, questions, answers, question_ids = [], [], [], []

    for group in squad_dict["data"]:
        for passage in group["paragraphs"]:
            context = passage["context"]
            for qa in passage["qas"]:
                question = qa["question"]

                if not incl_impossibles and "plausible_answers" in qa.keys():
                    continue

                # Check if we need to be extracting from 'answers' or 'plausible_answers'
                if "plausible_answers" in qa.keys():
                    access = "plausible_answers"
                else:
                    access = "answers"

                contexts.append(context)
                questions.append(question)
                answers.append(qa[access][0])
                question_ids.append(qa["id"])

    # Return formatted data lists
    return contexts, questions, answers, question_ids


def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer["text"]
        start_idx = answer["answer_start"]
        end_idx = start_idx + len(gold_text)

        # fix if answers are off by a character or two
        if context[start_idx:end_idx] == gold_text:
            answer["answer_end"] = end_idx

        elif context[start_idx - 1 : end_idx - 1] == gold_text:
            answer["answer_start"] = start_idx - 1
            answer["answer_end"] = (end_idx - 1)  # When the gold label is off by one character

        elif context[start_idx - 2 : end_idx - 2] == gold_text:
            answer["answer_start"] = start_idx - 2
            answer["answer_end"] = (end_idx - 2)  # When the gold label is off by two characters


def add_token_positions(tokenizer, encodings, answers):

    # Initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in range(len(answers)):

        # Append start/end token position using char_to_token method
        start_positions.append(encodings.char_to_token(i, answers[i]["answer_start"]))
        end_positions.append(encodings.char_to_token(i, answers[i]["answer_end"]))

        # If start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length

        # End position cannot be found, char_to_token found space, so shift position until found
        shift = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]["answer_end"] - shift)
            shift += 1

    # Update our encodings object with the new token-based start/end positions
    encodings.update({"start_positions": start_positions, "end_positions": end_positions})


def clean_up_answer(tokens, answer_start, answer_end):
    answer = tokens[answer_start]

    for i in range(answer_start + 1, answer_end + 1):
        
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        elif tokens[i] in [',', '.', ':', '?']:
            answer += tokens[i]
        
        else:
            answer += ' ' + tokens[i]

    return answer


def main(args):

    contexts, questions, answers, question_ids = read_dataset(f'{args.data_file}', args.incl_impossibles)
    
    add_end_idx(answers, contexts)

    model = AutoModelForQuestionAnswering.from_pretrained(f'{args.model}')
    tokenizer = AutoTokenizer.from_pretrained(f'{args.model}')

    encodings = tokenizer(contexts, questions, truncation=True, padding=True)

    add_token_positions(tokenizer, encodings, answers)

    dataset = SquadDataset(encodings)
    print("\nDataset initialized")

    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)
    model.eval()

    # Create a list of question id lists of size BATCH_SIZE to output also question ids
    question_id_batches = [question_ids[x: x + args.batch_size] for x in range(0, len(question_ids), args.batch_size)]

    # Initialize validation set data loader
    loader = DataLoader(dataset, batch_size=args.batch_size)

    json_dict = {}

    # loop through batches
    for question_ids, batch in zip(question_id_batches, loader):

        with torch.no_grad():
            # pull batched items from loader
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            token_lists = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

            outputs = model(input_ids, attention_mask=attention_mask)

            start_pred = torch.argmax(outputs['start_logits'], dim=1)
            end_pred = torch.argmax(outputs['end_logits'], dim=1)

            preds = zip(start_pred, end_pred)
            i = 0

            for tokens, (start, end) in zip(token_lists, preds):
                answer = clean_up_answer(tokens, start, end)
                print(f'"{question_ids[i]}": {json.dumps(answer)},')
                json_dict[question_ids[i]] = answer
                i += 1

    # Create a JSON file with the question IDs as keys and predicter answers as values
    with open(f'{args.out_file}', 'w', encoding='utf8') as out_file:
            json.dump(json_dict, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Output model predictions for evaluation')
    parser.add_argument('model', metavar='model_dir', help='Model to be used for predictions.')
    parser.add_argument('data_file', metavar='data.json', help='Input data JSON file.')
    parser.add_argument("--incl_impossibles", dest='incl_impossibles', action='store_true', required=False,
                        help="Generate predictions for also impossible questions.")
    parser.add_argument("--batch_size", required=False, type=int, choices=[8, 16, 32], default=8,
                        help="Batch size. Choose 8, 16, or 32")
    parser.add_argument("--out_file", required=False, default="predictions",type=str,
                        help="Name for the predictions json file.")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    main(args)
