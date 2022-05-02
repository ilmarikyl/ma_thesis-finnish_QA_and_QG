from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, re

tokenizer = AutoTokenizer.from_pretrained("Finnish-NLP/gpt2-medium-finnish")

model = AutoModelForCausalLM.from_pretrained("FNL-GPT2-newsplit-qa-v1-epoch1_aka_checkpoint-96906")

print('tokenizer:', tokenizer)
print('---')

model.eval()

c = '''Ulkomuodoltaan hylkeet ovat sileitä ja pulleita. Ruumiinrakenne soveltuu sulavaan vedessä liikkumiseen. Ranteesta ja kämmenestä ovat muodostuneet etuevät ja nilkasta ja jalkaterästä takaevät. Evät ovat heikot eikä niitä voi käyttää apuna maalla liikkumiseen. Hylkeet liikkuvatkin maalla siten, että ne siirtävät painoa rinnan ja vatsan varaan. Erotuksena lähisukulaisistaan korvahylkeistä, joihin kuuluvat muun muassa merileijonat, varsinaisilla hylkeillä ei ole ulkoisia korvalehtiä. Varsinaisten hylkeiden uiminen tapahtuu evien ja ruumiin takaosan sivuttaissuuntaista liikettä apuna käyttäen.'''

q = "Mihin hylkeiden evät eivät sovellu?"

input_context = f'Konteksti: {c}\nKysymys: {q}\nVastaus:'
input_ids = tokenizer(input_context, return_tensors="pt").input_ids

len_input_ids = len(tokenizer(input_context, return_tensors="pt").input_ids[0])

outputs = model.generate(input_ids=input_ids, max_length=len_input_ids + 50, num_return_sequences=1, do_sample=True)

print('---' * 20)
print('')
print('Generated outputs:\n')

outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for i, output in enumerate(outputs):
    print(f'#{i + 1}\n')
    print(output)
    print('---' * 20)
# ---

