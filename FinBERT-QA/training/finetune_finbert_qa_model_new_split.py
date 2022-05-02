import json, transformers, argparse, torch, sys

import  collections, json, os, re, string

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForQuestionAnswering, AdamW

class SquadDataset(torch.utils.data.Dataset):
	def __init__(self, encodings):
		self.encodings = encodings

	def __getitem__(self, idx):
		return {
			key: torch.tensor(val[idx]) for key, val in self.encodings.items()
		}

	def __len__(self):
		return len(self.encodings.input_ids)


def read_dataset(path, include_impossibles):
	# open JSON file and load intro dictionary
	with open(path, "rb") as f:
		squad_dict = json.load(f)

	# Initialize lists for contexts, questions, and answers
	contexts, questions, answers = [], [], []

	# Iterate through all data in squad data
	for group in squad_dict["data"]:
		for passage in group["paragraphs"]:
			context = passage["context"]
			for qa in passage["qas"]:
				question = qa["question"]

				# Ignore impossible questions unless "--incl_impossibles" argument was used 
				if qa["is_impossible"] and not include_impossibles:
					continue

				# Check if we need to be extracting from 'answers' or 'plausible_answers'
				if "plausible_answers" in qa.keys():
					access = "plausible_answers"
				else:
					access = "answers"

				for answer in qa[access]:
					contexts.append(context)
					questions.append(question)
					answers.append(answer)
					
	return contexts, questions, answers

def clean_up_answer(tokens, answer_start, answer_end):
	answer = tokens[answer_start]

	for i in range(answer_start + 1, answer_end + 1):
		
		# If it's a subword token, then recombine it with the previous token.
		if tokens[i][0:2] == '##':
			answer += tokens[i][2:]
		elif tokens[i] in [',', '.', ':', '?']:
			answer += tokens[i]
		
		# Otherwise, add a space then the token.
		else:
			answer += ' ' + tokens[i]

	return answer


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


def main(args):

	# Create the output dir if it doesn't yet exist
	Path(args.save_path).mkdir(parents=True, exist_ok=True)

	# Read datasets
	train_contexts, train_questions, train_answers = read_dataset("../../../datasets/qg_train_split-64604.json", args.incl_impossibles)
	val_contexts, val_questions, val_answers = read_dataset("../../../datasets/qg_dev_split-4902.json", args.incl_impossibles)

	# Append answer end markers
	add_end_idx(train_answers, train_contexts)
	add_end_idx(val_answers, val_contexts)

	# DEBUGGING --->
	# print(train_contexts[0])
	# print(train_questions[0])
	# print(train_answers[0])

	# for i, char in enumerate(train_contexts[3]):
	#     if (i >= train_answers[3]["answer_start"]and i <= train_answers[3]["answer_end"]):
	#         print(f"{i}: {char}")
	#  <------------

	# Load pre-trained model and tokenizer
	# model = AutoModelForQuestionAnswering.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
	tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")

	train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
	val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)


	print('train_encodings[0]:', tokenizer.convert_ids_to_tokens(train_encodings[0].ids))

	print('train_encodings[0]:', train_encodings[0].special_tokens_mask)


	# print(tokenizer.convert_ids_to_tokens(train_encodings[0]))


	exit()
	# DEBUGGING --->
	# print("Encodings created")
	# print(tokenizer.decode(train_encodings["input_ids"][0]))
	# print("\ntrain_encodings.keys():")
	# print(train_encodings.keys())
	# <------------

	# Add token positions to training data
	add_token_positions(tokenizer, train_encodings, train_answers)
	add_token_positions(tokenizer, val_encodings, val_answers)

	print("\nToken positions added.")
	# print(train_encodings.keys())

	# Build training dataset
	train_dataset = SquadDataset(train_encodings)
	print("\nTraining dataset initialized.")

	val_dataset = SquadDataset(val_encodings)
	print("\nValidation dataset initialized.")

	# Setup GPU/CPU
	device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

	# Move model over to detected device
	model.to(device)

	# Activate training mode of model
	model.train()

	# Initialize adam optimizer with weight decay (to reduce chance of overfitting)
	optimizer = AdamW(model.parameters(), lr=5e-5)

	# Initialize data loader for training data
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

	print(f'Epochs: {args.epochs}')
	print(f'Batch size: {args.batch_size}')
	

	for epoch in range(args.epochs):
		# Set model to train mode
		model.train()

		# Setup loop (we use tqdm for the progress bar)
		loop = tqdm(train_loader, leave=True)

		for i, batch in enumerate(loop):
			# print('batch number:', i)
			# print('len(loop):', len(loop))

			# Initialize calculated gradients (from prev step)
			optimizer.zero_grad()

			# Pull all the tensor batches required for training
			input_ids = batch["input_ids"].to(device)
			attention_mask = batch["attention_mask"].to(device)
			start_positions = batch["start_positions"].to(device)
			end_positions = batch["end_positions"].to(device)

			# Train model on batch and return outputs (incl. loss)
			outputs = model(
				input_ids,
				attention_mask=attention_mask,
				start_positions=start_positions,
				end_positions=end_positions,
			)

			# Extract loss
			loss = outputs[0]

			# Calculate loss for every parameter that needs grad update
			loss.backward()

			# Update parameters
			optimizer.step()

			# Print relevant info to progress bar
			loop.set_description(f"Epoch {epoch}")
			loop.set_postfix(loss=loss.item())

			if i == round(len(loop)) / 2 or i == len(loop)-1:

				# switch model out of training mode
				model.eval()
				# initialize validation set data loader
				val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
				# initialize list to store accuracies
				acc = []
				# loop through batches
				for batch in val_loader:
					# we don't need to calculate gradients as we're not training
					with torch.no_grad():
						# pull batched items from loader
						input_ids = batch['input_ids'].to(device)
						attention_mask = batch['attention_mask'].to(device)
						# we will use true positions for accuracy calc
						start_true = batch['start_positions'].to(device)
						end_true = batch['end_positions'].to(device)
						# make predictions
						outputs = model(input_ids, attention_mask=attention_mask)
						# pull prediction tensors out and argmax to get predicted tokens
						start_pred = torch.argmax(outputs['start_logits'], dim=1)
						end_pred = torch.argmax(outputs['end_logits'], dim=1)
						# calculate accuracy for both and append to accuracy list
						acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
						acc.append(((end_pred == end_true).sum()/len(end_pred)).item())
				# calculate average accuracy in total
				acc = sum(acc)/len(acc)
				print(f'VALIDATION for Epoch {epoch}, batch # {i} - Exact match (EM): {acc}')

				with open(f'{args.save_path.split("/")[-1]}-train-eva-results.txt', 'a') as eval_file:
					eval_file.write(f'VALIDATION for Epoch {epoch}, batch # {i} - Exact match (EM): {acc}\n')


				model.train()

		print(f"Epoch #{epoch + 1} done.")
		with open(f'{args.save_path.split("/")[-1]}-train-eval-results.txt', 'a') as eval_file:
			eval_file.write(f'VALIDATION for Epoch {epoch}, batch # {i} - Exact match (EM): {acc}\n')


		# Save a checkpoint model (unless all epochs are completed)
		if epoch + 1 < args.epochs:
			# Create the output dir if it doesn't yet exist
			Path(f'{args.save_path}-cp-epoch-{epoch+1}').mkdir(parents=True, exist_ok=True)
			model.save_pretrained(f'{args.save_path}-cp-epoch-{epoch+1}')
			tokenizer.save_pretrained(f'{args.save_path}-cp-epoch-{epoch+1}')

	# Format: "parent_dir/sub_dir"
	model.save_pretrained(args.save_path)
	tokenizer.save_pretrained(args.save_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Script for fine-tuning a FinBERT model for question answering')
	parser.add_argument("--epochs", required=True, type=int, choices=range(1,21), metavar="[1-20]",
						help="How many epoch the model will be trained. Choose between 1 and 20")
	parser.add_argument("--batch_size", required=False, type=int, choices=[8, 16, 32], default=8,
						help="Batch size. Choose 8, 16, or 32")
	parser.add_argument("--incl_impossibles", dest='incl_impossibles', action='store_true', required=False,
						help="Use also impossible questions for fine-tuning")
	parser.add_argument("--save_path", required=True, type=str, help="Path for saving the model")

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)

	args = parser.parse_args()

	main(args)
