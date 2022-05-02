from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, re

tokenizer = AutoTokenizer.from_pretrained("../FinGPT-QG/gpt_base_model")
model = AutoModelForCausalLM.from_pretrained("checkpoint-60000")

model.eval()


c = '''
Helsingin yliopisto (lyhenne HY; ruots. Helsingfors universitet) on Suomen suurin ja vanhin tiedekorkeakoulu.

Helsingin yliopistossa on noin 35 000 opiskelijaa sekä lähes 4 000 tutkijaa ja opettajaa. Tohtoreita Helsingin yliopistosta valmistuu vuosittain noin 450.

Yliopiston toimintaa on 1990-luvulta lähtien keskitetty neljälle kampukselle: keskustaan, Kumpulaan, Meilahteen ja Viikkiin. Lisäksi yliopistoon kuuluu useita tutkimuslaitoksia ja yksiköitä ympäri maan. Helsingin yliopisto on Suomen ainut kaksikielinen tiedeyliopisto, eli sen tutkintokielet ovat suomi ja ruotsi.

Kansainvälisessä Shanghain yliopistovertailussa Helsingin yliopisto sijoittui vuonna 2017 sijalle 56 ja sijoittui ainoana yliopistona Suomesta maailman sadan parhaan yliopiston joukkoon. Times Higher Education -luokituksessa Helsingin yliopisto ylsi vuonna 2017 Euroopan 31. parhaaksi yliopistoksi.

Helsingin yliopisto kuuluu ainoana suomalaisyliopistona kansainväliseen Euroopan tutkimusyliopistojen liittoon.
'''

q = "Mitkä ovat yliopiston kampukset?"

model_input = f'Konteksti: {c}\nKysymys: {q}\n'

input_ids = tokenizer(model_input, return_tensors="pt").input_ids


outputs = model.generate(input_ids=input_ids, max_length=400, num_return_sequences=2, do_sample=True)

# DEBUG
# print('tokenizer:', tokenizer)
# print('input_ids:', input_ids)
# print('Outputs (ids):', outputs)

print('---' * 10)
print(f'Input passage: "{c}"')
print('')
print('Generated outputs:\n')

outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for i, output in enumerate(outputs):
    print(f'#{i + 1}\n')
    print(output.split("Vastaus: ")[1])
    print('---' * 10)
# ---