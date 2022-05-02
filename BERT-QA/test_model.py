import json, torch, transformers
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from tokenizers import Encoding as EncodingFast
from transformers import DistilBertTokenizerFast
from transformers import AutoModel, AutoTokenizer, AutoModelForQuestionAnswering

model_path = 'models/finBERT-QA-newsplit-v1-cp-epoch-2' 
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
questions = []

passage = '''
Helsingin yliopisto (lyhenne HY; ruots. Helsingfors universitet) on Suomen suurin ja vanhin tiedekorkeakoulu.

Helsingin yliopistossa on noin 35 000 opiskelijaa sekä lähes 4 000 tutkijaa ja opettajaa. Tohtoreita Helsingin yliopistosta valmistuu vuosittain noin 450.

Yliopiston toimintaa on 1990-luvulta lähtien keskitetty neljälle kampukselle: keskustaan, Kumpulaan, Meilahteen ja Viikkiin. Lisäksi yliopistoon kuuluu useita tutkimuslaitoksia ja yksiköitä ympäri maan. Helsingin yliopisto on Suomen ainut kaksikielinen tiedeyliopisto, eli sen tutkintokielet ovat suomi ja ruotsi.

Kansainvälisessä Shanghain yliopistovertailussa Helsingin yliopisto sijoittui vuonna 2017 sijalle 56 ja sijoittui ainoana yliopistona Suomesta maailman sadan parhaan yliopiston joukkoon. Times Higher Education -luokituksessa Helsingin yliopisto ylsi vuonna 2017 Euroopan 31. parhaaksi yliopistoksi.

Helsingin yliopisto kuuluu ainoana suomalaisyliopistona kansainväliseen Euroopan tutkimusyliopistojen liittoon.
'''

while True:
    question = input('Syötä kysymys: ')

    if question == 'q':
        break

    questions.append(question)


answers = []
question = ''

for question in questions:
    model.eval()
    input_ids = tokenizer.encode(passage, question)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print('TOKENS:', tokens)

    sep_index = input_ids.index(tokenizer.sep_token_id)

    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                    token_type_ids=torch.tensor([segment_ids]), return_dict=False) # The segment IDs to differentiate question from answer_text


    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Combine the tokens in the answer and print it out.
    answer = ' '.join(tokens[answer_start:answer_end+1])

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        elif tokens[i] in [',', '.', ':', '?']:
            answer += tokens[i]
        
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]


    answers.append(answer)

    print(f'Document:\n"{passage}"\n')

    for question, answer in zip(questions, answers):
        print('---'*10)
        print(f'Question:\n "{question}"')
        print(f'\nModel output answer:\n "{answer}"')

def clean_up_answer(tokens, answer_start, answer_end):
    answer = tokens[answer_start]

    for i in range(answer_start + 1, answer_end + 1):
        
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        elif tokens[i] in [',', '.', ':', '?']:
            answer += tokens[i]
        
        else:
            answer += ' ' + tokens[i]

    return answer
