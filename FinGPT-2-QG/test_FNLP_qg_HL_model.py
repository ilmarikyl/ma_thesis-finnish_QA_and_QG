from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, re


# config = AutoConfig.from_pretrained('words-checkpoint-7336')
tokenizer = AutoTokenizer.from_pretrained("Finnish-NLP/gpt2-medium-finnish")

model = AutoModelForCausalLM.from_pretrained("FNLP-gpt-qg-HL_v2_checkpoint-64604")

print('tokenizer:', tokenizer)
# print('model:', model)

# print('---')
# print(config)
# print('---')
# print('model:', model)
print('---')

model.eval()

# Pipeline Test
# text_generator = pipeline('text-generation',model='gpt_base_model', tokenizer='gpt_base_model')
# result = text_generator('Olipa kerran ')[0]['generated_text']
# print('Results:', result)

# Encoding and decoding test

# input_ids = tokenizer.encode(test_input)
# print('input_ids (after encoding):', input_ids)
# print('input_ids decoded:', tokenizer.decode(input_ids))
# print('---')


# ---

c = '''Jättiläismetsäkarju eli jättiläismetsäsika eli jättisika (Hylochoerus meinertzhageni) on keskisen ja läntisen Afrikan metsissä elävä elinvoimainen sorkkaeläinlaji. Se on sukunsa Hylochoerus ainoa laji. Jättiläismetsäkarjut ovat suurimpia luonnonvaraisia sikoja. Ne voivat kasvaa jopa 210 senttimetriä pitkiksi ja painaa 275 kilogrammaa. Niiden ruumis on tanakka ja pää leveä, mutta jalat ovat lyhyet. Nahkaa peittävät pitkät ja karkeat karvat, jotka nousevat pystyyn eläimen kiihtyessä.'''



a = "210 senttimetriä"

answer_start = c.index(a)

hl_passage = f'{c[:answer_start]}[HL]{a}[HL]{c[len(a) + answer_start:]}'

# print(hl_passage)
# exit()



# ---
input_context = f'<|startoftext|>Konteksti: {hl_passage}\nVastaus: {a}\nKysymys:'
input_ids = tokenizer(input_context, return_tensors="pt").input_ids
# print('input_ids:', input_ids)

len_input_ids = len(tokenizer(input_context, return_tensors="pt").input_ids[0])

outputs = model.generate(input_ids=input_ids, max_length=len_input_ids + 30, num_return_sequences=1, do_sample=True)
# print('Outputs (ids):', outputs)

print('---' * 20)
# print(f'Input passage: "{input_context}"')
print('')
print('Generated outputs:\n')

outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for i, output in enumerate(outputs):
    print(f'#{i + 1}\n')
    # cleaned = re.sub(r'(\.)(.*?)$',  r'\1', output)
    print(output)
    print('---' * 20)
# ---

# alla experimentaaliset

# ----
outputs = model.generate(input_ids=input_ids, max_length=len_input_ids + 30, num_beams=3, early_stopping=True)
# print('Outputs (ids):', outputs)

print('---' * 20)
# print(f'Input passage: "{input_context}"')
print('')
print('Generated outputs2:\n')

outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for i, output in enumerate(outputs):
    print(f'#{i + 1}\n')
    # cleaned = re.sub(r'(\.)(.*?)$',  r'\1', output)
    print(output)
    print('---' * 20)


# -----
outputs = model.generate(input_ids=input_ids, max_length=len_input_ids + 30, do_sample=True, top_k=50, top_p=0.95)
# print('Outputs (ids):', outputs)

print('---' * 20)
# print(f'Input passage: "{input_context}"')
print('')
print('Generated outputs3:\n')

outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for i, output in enumerate(outputs):
    print(f'#{i + 1}\n')
    # cleaned = re.sub(r'(\.)(.*?)$',  r'\1', output)
    print(output)
    print('---' * 20)




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