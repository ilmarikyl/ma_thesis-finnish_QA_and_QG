from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, re


# config = AutoConfig.from_pretrained('words-checkpoint-7336')
tokenizer = AutoTokenizer.from_pretrained("gpt_base_model")

model = AutoModelForCausalLM.from_pretrained("NAQG_checkpoint-8076")

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

c = '''
Helsingin yliopisto (lyhenne HY; ruots. Helsingfors universitet) on Suomen suurin ja vanhin tiedekorkeakoulu.

Helsingin yliopistossa on noin 35 000 opiskelijaa sekä lähes 4 000 tutkijaa ja opettajaa. Tohtoreita Helsingin yliopistosta valmistuu vuosittain noin 450.

Yliopiston toimintaa on 1990-luvulta lähtien keskitetty neljälle kampukselle: keskustaan, Kumpulaan, Meilahteen ja Viikkiin. Lisäksi yliopistoon kuuluu useita tutkimuslaitoksia ja yksiköitä ympäri maan. Helsingin yliopisto on Suomen ainut kaksikielinen tiedeyliopisto, eli sen tutkintokielet ovat suomi ja ruotsi.

Kansainvälisessä Shanghain yliopistovertailussa Helsingin yliopisto sijoittui vuonna 2017 sijalle 56 ja sijoittui ainoana yliopistona Suomesta maailman sadan parhaan yliopiston joukkoon. Times Higher Education -luokituksessa Helsingin yliopisto ylsi vuonna 2017 Euroopan 31. parhaaksi yliopistoksi.

Helsingin yliopisto kuuluu ainoana suomalaisyliopistona kansainväliseen Euroopan tutkimusyliopistojen liittoon.
'''




# ---
input_context = f'Konteksti: {c}\nKysymys: '
input_ids = tokenizer(input_context, return_tensors="pt").input_ids
# print('input_ids:', input_ids)

outputs = model.generate(input_ids=input_ids, max_length=350, num_return_sequences=6, do_sample=True)
# print('Outputs (ids):', outputs)

print("\n\nKonteksti:", c)
print('')
print('---' * 20)
print('')
print('Generoidut kysymykset:\n')

outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)


for i, output in enumerate(outputs):
    print(f'#{i + 1}\n')
    # cleaned = re.sub(r'(\.)(.*?)$',  r'\1', output)
    print(output.split("Kysymys: ")[1])
    print('---' * 20)
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