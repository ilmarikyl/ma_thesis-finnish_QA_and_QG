import json, transformers, argparse, torch, sys

import  collections, json, os, re, string

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForQuestionAnswering, AdamW, AutoModelForCausalLM


def main():


	# Load pre-trained model and tokenizer
	# model = AutoModelForQuestionAnswering.from_pretrained("bert-base-multilingual-cased")
	model = AutoModelForCausalLM.from_pretrained("Finnish-NLP/gpt2-medium-finnish")
	# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


	print('MODEL')
	print(model)

if __name__ == '__main__':
	main()
