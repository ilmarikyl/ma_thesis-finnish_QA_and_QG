# -*- coding: utf-8 -*-
from transformers import pipeline

nlp_qa = pipeline(
    'question-answering',
    model='models/M-BERT-QA_newsplit-v1-cp-epoch-2',
    tokenizer='models/M-BERT-QA_newsplit-v1-cp-epoch-2'
)

passage = '''Ulkomuodoltaan hylkeet ovat sileitä ja pulleita. Ruumiinrakenne soveltuu sulavaan vedessä liikkumiseen. Ranteesta ja kämmenestä ovat muodostuneet etuevät ja nilkasta ja jalkaterästä takaevät. Evät ovat heikot eikä niitä voi käyttää apuna maalla liikkumiseen. Hylkeet liikkuvatkin maalla siten, että ne siirtävät painoa rinnan ja vatsan varaan. Erotuksena lähisukulaisistaan korvahylkeistä, joihin kuuluvat muun muassa merileijonat, varsinaisilla hylkeillä ei ole ulkoisia korvalehtiä. Varsinaisten hylkeiden uiminen tapahtuu evien ja ruumiin takaosan sivuttaissuuntaista liikettä apuna käyttäen.'''

questions = []
usr_input = ""

while True:
    usr_input = input("Syötä kysymys: ")

    if usr_input == 'q':
        break
    
    questions.append(usr_input) 

print('')
for q in questions:

    ans = nlp_qa(
        {
            'question': q,
            'context': passage
        }
    )

    print('Kysymys:', q)
    print('Vastaus:', ans["answer"])
    print('---'*10)