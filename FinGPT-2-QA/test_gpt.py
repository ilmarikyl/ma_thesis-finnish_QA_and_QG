from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, re


config = AutoConfig.from_pretrained('gpt_base_model')
tokenizer = AutoTokenizer.from_pretrained("gpt_base_model")
model = AutoModelForCausalLM.from_pretrained("gpt_base_model")

print('tokenizer:', tokenizer)

print('---')
print(config)
print('---')

model.eval()


# Encoding and decoding test
test_sentence = "<bos> <fgn> <frml> <kgb> Tämä on testilause.  <eos> <pad> <pad><pad>"
input_ids = tokenizer.encode(test_sentence)
print('input_ids (after encoding):', input_ids)
print('input_ids decoded:', tokenizer.decode(input_ids, skip_special_tokens=True))
print('---')


print('tokenizer.bos_token_id:', tokenizer.bos_token_id)
print('tokenizer.eos_token_id:', tokenizer.eos_token_id)

tokens = tokenizer.convert_ids_to_tokens(input_ids)

# ---
input_context = "Viikkopassit on noudettavissa Esport Centerin ja Esport Bristolin asiakasneuvojilta torstaina 25.11.2021.  Viikkopassi on aktivoitava 31.12.2021 mennessä."
input_ids = tokenizer(input_context, return_tensors="pt").input_ids
print('input_ids:', input_ids)

outputs = model.generate(input_ids=input_ids, max_length=120, num_return_sequences=3, do_sample=True)
print('Outputs (ids):', outputs)

print('---' * 10)
print(f'Input passage: "{input_context}"')
print('')
print('Generated outputs:\n')

outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for i, output in enumerate(outputs):
    print(f'#{i + 1}\n')
    print(output)
    print('---' * 10)