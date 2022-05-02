from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, re


config = AutoConfig.from_pretrained("Finnish-NLP/gpt2-medium-finnish")
tokenizer = AutoTokenizer.from_pretrained("Finnish-NLP/gpt2-medium-finnish")

model = AutoModelForCausalLM.from_pretrained("Finnish-NLP/gpt2-medium-finnish")

print('tokenizer:', tokenizer)
print('------------------------')
print('model:', model)

print('---\nCONFIG')
print(config)
# print('---')
# print('model:', model)
print('---')

model.eval()

# Pipeline Test
# text_generator = pipeline('text-generation',model='gpt_base_model', tokenizer='gpt_base_model')
# result = text_generator('Olipa kerran ')[0]['generated_text']
# print('Results:', result)

# Encoding and decoding test
test_sentence = "<bos> <fgn> <frml> <kgb> Tämä on testilause.  <eos> <pad> <pad><pad>"
input_ids = tokenizer.encode(test_sentence)
print('input_ids (after encoding):', input_ids)
print('input_ids decoded:', tokenizer.decode(input_ids, skip_special_tokens=True))
print('---')


print('tokenizer.bos_token_id:', tokenizer.bos_token_id)
print('tokenizer.eos_token_id:', tokenizer.eos_token_id)



# BERT only needs the token IDs, but for the purpose of inspecting the 
# tokenizer's behavior, let's also get the token strings and display them.
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# For each token and its id...
# for token, id in zip(tokens, input_ids):
    
#     # If this is the [SEP] token, add some space around it to make it stand out.
#     if id == tokenizer.sep_token_id:
#         print('')
    
#     # Print the token string and its ID in two columns.
#     print('{:<12} {:>6,}'.format(token, id))

#     if id == tokenizer.sep_token_id:
#         print('')
# ---

# ---
# input_context = "Minun piti kirjoittaa gradua,"
input_context ='''
I have a dog = Minulla on koira.
I speak English = Minä puhun englantia.
What is your name ='''
input_ids = tokenizer(input_context, return_tensors="pt").input_ids
print('input_ids:', input_ids)

outputs = model.generate(input_ids=input_ids, max_length=65, num_return_sequences=5, do_sample=True)
print('Outputs (ids):', outputs)

print('---' * 10)
print(f'Input passage: "{input_context}"')
print('')
print('Generated outputs:\n')

outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for i, output in enumerate(outputs):
    print(f'#{i + 1}\n')
    # cleaned = re.sub(r'(\.)(.*?)$',  r'\1', output)
    print(output)
    print('---' * 10)
# ---



# print('---')

# print('')
# print("Encodingsit luotu")
# print('train_encodings:', train_encodings)


# print('Decoded back:')
# print(tokenizer.decode(train_encodings["input_ids"][1]))
# print("")


# print('')
# print("train_encodings.keys()")
# print(train_encodings.keys())



# start_scores, end_scores = model(torch.tensor([input_ids]), return_dict=False) # The segment IDs to differentiate question from answer_text