from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, re, json, argparse, sys


def read_dataset(path, incl_impossibles=False):
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

    return contexts, questions, answers, question_ids


def main(args):

    tokenizer = AutoTokenizer.from_pretrained("Finnish-NLP/gpt2-large-finnish", bos_token='<|startoftext|>')
    tokenizer.add_special_tokens({'additional_special_tokens': ['[HL]']})

    # Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
    model = AutoModelForCausalLM.from_pretrained(f"{args.model}", pad_token_id=tokenizer.eos_token_id)

    # Resize the token embeddings because we've just added new tokens 
    model.resize_token_embeddings(len(tokenizer))

    generator = pipeline('text-generation', tokenizer=tokenizer, model=model)

    contexts, questions, answers, question_ids = read_dataset(f'{args.data_file}', args.incl_impossibles)

    output_dict = {"data": []}
    i = 1

    for c, q, a, q_id in zip(contexts, questions, answers, question_ids):

        answer_text = a["text"]
        answer_start = a["answer_start"]

        hl_passage = f'{c[:answer_start]}[HL]{answer_text}[HL]{c[len(answer_text) + answer_start:]}'

        item = {"id": q_id, "reference": q}
        model_input = f'<|startoftext|>Konteksti: {hl_passage}\nVastaus: {answer_text}\n'
        # print('model_input:', model_input)
        len_input_ids = len(tokenizer(model_input, return_tensors="pt").input_ids[0])
        output = generator(model_input, max_length=len_input_ids + 30)
        try:
            hypothesis = output[0]["generated_text"].split("Kysymys: ")[1]
        except:
            hypothesis = "ERROR"
        item["hypothesis"] = hypothesis
        print(f'{i} / {len(question_ids)} - {item}')

        output_dict["data"].append(item)
        i += 1


    with open(f'{args.model}_preds.json', 'w',encoding='utf8') as out_f:
        out_f.write(json.dumps(output_dict))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Output model predictions for evaluation')
    parser.add_argument('model', metavar='model_dir', help='Model to be used for predictions.')
    parser.add_argument('--data_file', metavar='data.json', default='qg_test_split-4822.json',
                        required=False, help='Input data JSON file.')
    parser.add_argument("--incl_impossibles", dest='incl_impossibles', action='store_true', required=False,
                        help="Generate predictions for also impossible questions.")
    

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    main(args)