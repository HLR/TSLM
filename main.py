import sys
sys.path.append("..")

import json
import spacy
import torch
from src.single_transformer import AttentionWide
from src.models import LSTMFlair, FCGumble, FullyConnected
# from torch_geometric.nn import AGNNConv
# from torch_geometric.data import Data
import torch.nn.functional as F
import numpy
import difflib
import random


from transformers import BertTokenizer, BertModel, BertTokenizerFast, RobertaTokenizerFast, RobertaTokenizer, AutoTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
tokenizer_fast = BertTokenizerFast.from_pretrained('bert-large-uncased')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
roberta_tokenizer_fast = RobertaTokenizerFast.from_pretrained('roberta-large')


with open('data/train_propara_roberta_version.json') as f:
    train_propara_roberta_qa = json.load(f)
with open('data/train_propara.json') as f:
    train_propara_qa = json.load(f)
with open('data/test_propara_roberta_version.json') as f:
    test_propara_roberta_qa = json.load(f)
with open('data/test_propara.json') as f:
    test_propara_qa = json.load(f)
with open('data/dev_propara_roberta_version.json') as f:
    dev_propara_roberta_qa = json.load(f)
with open('data/dev_propara.json') as f:
    dev_propara_qa = json.load(f)
    
    
def extract_timestamp(inputs, time):
    timestamp_id = []
    if time == -1:
        check = -1
        for index, ids in enumerate(inputs['input_ids'][0]):
            if ids == 102:
                check += 1
            if inputs['token_type_ids'][0][index] == 0:
                timestamp_id.append(0)
            else:
                timestamp_id.append(2)
    else:
        check = -1
        for index, ids in enumerate(inputs['input_ids'][0]):
            if ids == 102:
                check += 1
            if inputs['token_type_ids'][0][index] == 0:
                timestamp_id.append(0)
            else:
                if check < time :
                    timestamp_id.append(1)
                elif check == time:
                    timestamp_id.append(2)
                else:
                    timestamp_id.append(3)
    timestamp_id = torch.tensor(timestamp_id).to(device=inputs['input_ids'].device)
    inputs['timestep_type_ids'] = timestamp_id.unsqueeze(0)
    return inputs

from stemming.porter2 import stem

def location_match(p_loc, g_loc):
    if p_loc == g_loc:
        return True

    p_string = ' %s ' % ' '.join([stem(x) for x in p_loc.lower().replace('"','').split()])
    g_string = ' %s ' % ' '.join([stem(x) for x in g_loc.lower().replace('"','').split()])

    if p_string in g_string:
        #print ("%s === %s" % (p_loc, g_loc))
        return True

    return False

def extract_timestamp_sequence(inputs, end_time):
    f_out = []
    for time in range(-1, end_time - 1):
        timestamp_id = []
        if time == -1:
            check = -1
            for index, ids in enumerate(inputs['input_ids'][time + 1]):
                if ids == 102:
                    check += 1
                if inputs['token_type_ids'][0][index] == 0:
                    timestamp_id.append(0)
                else:
                    timestamp_id.append(2)
        else:
            check = -1
            for index, ids in enumerate(inputs['input_ids'][time + 1]):
                if ids == 102:
                    check += 1
                if inputs['token_type_ids'][0][index] == 0:
                    timestamp_id.append(0)
                else:
                    if check < time :
                        timestamp_id.append(1)
                    elif check == time:
                        timestamp_id.append(2)
                    else:
                        timestamp_id.append(3)
        timestamp_id = torch.tensor(timestamp_id).to(device=inputs['input_ids'].device)
        f_out.append(timestamp_id)
    inputs['timestep_type_ids'] = torch.stack(f_out)
    return inputs

def roberta_extract_timestamp_sequence(inputs, end_time):
    f_out = []
    padding = 0
    for time in range(-1, end_time - 1):
        timestamp_id = []
        if time == -1:
            check = -1
            for index, ids in enumerate(inputs['input_ids'][time + 1]):
                if ids == 2:
                    check += 1
                    if check == 0:
                        padding = index + 1
                if check == -1:
                    timestamp_id.append(0)
                elif ids == 2:
                    timestamp_id.append(0)
                else:
                    timestamp_id.append(2)
        else:
            check = -1
            for index, ids in enumerate(inputs['input_ids'][time + 1]):
                if ids == 2:
                    check += 1
                if check == -1:
                    timestamp_id.append(0)
                elif ids == 2:
                    timestamp_id.append(0)
                else:
                    if check < time :
                        timestamp_id.append(1)
                    elif check == time:
                        timestamp_id.append(2)
                    else:
                        timestamp_id.append(3)
        timestamp_id = torch.tensor(timestamp_id).to(device=inputs['input_ids'].device)
        f_out.append(timestamp_id)
    inputs['timestep_type_ids'] = torch.stack(f_out)
    return inputs, padding

def extract_timestamp_sequence_cls(inputs, end_time):
    f_out = []
    for time in range(-1, end_time - 1):
        timestamp_id = []
        if time == -1:
            check = -1
            for index, ids in enumerate(inputs['input_ids'][time + 1]):
                if ids == 102:
                    check += 1
                if inputs['token_type_ids'][0][index] == 0:
                    timestamp_id.append(0)
                else:
                    timestamp_id.append(2)
        else:
            check = -1
            for index, ids in enumerate(inputs['input_ids'][time + 1]):
                if ids == 102:
                    check += 1
                    
                if ids == 102:
                    timestamp_id.append(0)
                elif inputs['token_type_ids'][0][index] == 0:
                    timestamp_id.append(0)
                else:
                    if ids == 101:
                        timestamp_id.append(2)
                    elif check < time :
                        timestamp_id.append(1)
                    elif check == time:
                        timestamp_id.append(2)
                    else:
                        timestamp_id.append(3)
        timestamp_id = torch.tensor(timestamp_id).to(device=inputs['input_ids'].device)
        f_out.append(timestamp_id)
    inputs['timestep_type_ids'] = torch.stack(f_out)
    return inputs

def make_entry(qa_classifier, change_classifier, spans, entity, para_id):
    data = []
    for step in range(1,len(qa_classifier)):
        item = {}
        item['para_id'] = para_id
        item['step'] = step
        item['entity'] = entity
        if step == 1:
            if qa_classifier[step-1] == 2:
                item['before'] = 'null'
            elif qa_classifier[step-1] == 1:
                item['before'] = 'unk'
            elif qa_classifier[step-1] == 0:
                item['before'] = spans[step-1]    
        else:
            item['before'] = data[step-2]['after']

        if qa_classifier[step] == 2:
            item['after'] = 'null'
        elif qa_classifier[step] == 1:
            item['after'] = "unk"
        elif qa_classifier[step] == 0:
            if qa_classifier[step-1] == 0:
                if change_classifier[step] == 0:
                    item['after'] = item['before']
                else:
                    if not spans[step]:
                        spans[step] = 'unk'
                    item['after'] = spans[step]
            else:
                if not spans[step]:
                    spans[step] = 'unk'
                item['after'] = spans[step]
        data.append(item)
    return data

def make_states(qa_classifier, change_classifier=None, spans=None):
    item = []
    create = 0
    destroy = 0
    for step in range(1,len(qa_classifier)):
        if step == 1:
            if qa_classifier[step-1] == 2:
                item.append('-')
            elif qa_classifier[step-1] == 1:
                item.append('?')
            elif qa_classifier[step-1] == 0:
                if not spans[step-1]:
                    spans[step-1] = "?"
                item.append(spans[step-1])

        if qa_classifier[step] == 2:
            if item[-1] != "-":
                if destroy == 0:
                    destroy = 1
                    item.append('-')
                else:
                    item.append(item[-1])
            else:
                item.append('-')
        elif qa_classifier[step] == 1:
            if item[-1] != "-":
                item.append('?')
            else:
                if create == 0:
                    create = 1
                    item.append('?')
                else:
                    item.append(item[-1])
        elif qa_classifier[step] == 0:
            if item[-1] != "-":
                if not spans[step]:
                    spans[step] = "?"
                item.append(spans[step])
            else:
                if create == 0:
                    create = 1
                    if not spans[step]:
                        spans[step] = "?"
                    item.append(spans[step])
                else:
                    item.append(item[-1])
    return item

def make_eval_entry(qa_classifier, spans, entity, para_id):
    data = []
    for step in range(1,len(qa_classifier)):
        item = {}
        item['para_id'] = para_id
        item['step'] = step
        item['entity'] = entity
        if step == 1:
            if qa_classifier[step-1] == 2:
                item['before'] = '-'
            elif qa_classifier[step-1] == 1:
                item['before'] = '?'
            elif qa_classifier[step-1] == 0:
                if not spans[step - 1]:
                    spans[step - 1] = 'Some Location'
                item['before'] = spans[step-1]    
        else:
            item['before'] = data[step-2]['after']

        if qa_classifier[step] == 2:
            item['after'] = '-'
        elif qa_classifier[step] == 1:
            item['after'] = "?"
        elif qa_classifier[step] == 0:
            if not spans[step]:
                spans[step] = 'Some Location'
            item['after'] = spans[step]
            
        if item['before'] == "-":
            if item['after'] == "-":
                item['action'] = "NONE"
            else:
                item['action'] = "CREATE"
        elif item['before'] == "?":
            if item['after'] == "-":
                item['action'] = "DESTROY"
            elif item['after'] == "?":
                item['action'] = "NONE"
            else:
                item['action'] = "MOVE"
        else:
            if item['after'] == "-":
                item['action'] = "DESTROY"
            elif item['after'] == "?":
                item['action'] = "MOVE"
            else:
                if item['after'] == item['before']:
                    item['action'] = "NONE"
                else:
                    item['action'] = "MOVE"
                    
        data.append(item)
    return data


def make_eval_entry_logical(qa_classifier, spans, entity, para_id):
    data = []
    destroy = 0
    created = 0
    for step in range(1,len(qa_classifier)):
        item = {}
        item['para_id'] = para_id
        item['step'] = step
        item['entity'] = entity
        if step == 1:
            if qa_classifier[step-1] == 2:
                item['before'] = '-'
            elif qa_classifier[step-1] == 1:
                item['before'] = '?'
            elif qa_classifier[step-1] == 0:
                if not spans[step - 1]:
                    spans[step - 1] = 'Some Location'
                item['before'] = spans[step-1]    
        else:
            item['before'] = data[step-2]['after']

        if qa_classifier[step] == 2:
            item['after'] = '-'
        elif qa_classifier[step] == 1:
            item['after'] = "?"
        elif qa_classifier[step] == 0:
            if not spans[step]:
                spans[step] = 'Some Location'
            item['after'] = spans[step]
            
        if item['before'] == "-":
            if item['after'] == "-":
                item['action'] = "NONE"
            else:
                if created == 0 and destroy == 0:
                    created = 1
                    item['action'] = "CREATE"
                else:
                    item['action'] = "NONE"
                    item['after'] = item['before']
        elif item['before'] == "?":
            if item['after'] == "-":
                if destroy == 0:
                    destroy = 1
                    item['action'] = "DESTROY"
                else:
                    item['action'] = "NONE"
                    item['after'] = item['before']
            elif item['after'] == "?":
                item['action'] = "NONE"
            else:
                item['action'] = "MOVE"
        else:
            if item['after'] == "-":
                if destroy == 0:
                    destroy = 1
                    item['action'] = "DESTROY"
                else:
                    if spans[step] != spans[step-1]:
                        item['action'] = "MOVE"
                        item['after'] = spans[step]
                    else:
                        item['action'] = "NONE"
                        item['after'] = item['before']
            elif item['after'] == "?":
                item['action'] = "MOVE"
            else:
                if item['after'] == item['before']:
                    item['action'] = "NONE"
                else:
                    item['action'] = "MOVE"
                    
        data.append(item)
    return data

from bert import BertForQuestionAnswering, ProceduralQABert, ProceduralQABertTest
from evaluation import get_metrics
from roberta import RobertaProceduralQA, RobertaProceduralQATest
from tqdm import tqdm


# model = RobertaProceduralQATest.from_pretrained('tli8hf/unqover-roberta-large-squad', return_dict=True)
model = RobertaProceduralQA.from_pretrained('tli8hf/unqover-roberta-large-squad', return_dict=True)

def test_model(model, test_set, tokenizer_fast, tokenizer, name="test", it=0):
    iter_data = []
    it_test_location_total = 0
    it_test_location_correct = 0
    it_test_status_total = {"-": 0, "?": 0, "Location": 0}
    it_test_status_correct = {"-": 0, "?": 0, "Location": 0}
    it_change_total = {"NC": 0, "M": 0, "C": 0, "D": 0}
    it_change_correct = {"NC": 0, "M": 0, "C": 0, "D": 0}
    data = []
    data_logical = []
    table = {}
    for sample_id in tqdm(range(len(test_set))):
        sample_data = {}
        try:
            sample = test_set[sample_id]
            sample_data['id'] = sample['para_id']
            sample_data['entities'] = sample['participants']
            sample_data['steps'] = sample['sentence_texts']
            sample_data['entity_step'] = []
            table_entry = {'para_id': sample['para_id'], 'entities': {}}
            
            story = sample['sentence_paragraph']
#             print("story is : ", story)
            participants = sample['participants']
        #     print(story)
            location_total = 0
            location_correct = 0
            status_total = {"-": 0, "?": 0, "Location": 0}
            status_correct = {"-": 0, "?": 0, "Location": 0}
            change_total = {"NC": 0, "M": 0, "C": 0, "D": 0}
            change_correct = {"NC": 0, "M": 0, "C": 0, "D": 0}
            for entity_id, states in enumerate(sample['states_annotation']):
                #The normal Scenario:
#                 question = "Where is " + str(participants[entity_id]) + "?!</s>"
#                 qa_stories = [question + story] * len(states)
                # The step scenario
                qa_stories = []
                for state_num, state in enumerate(states):
                    question = "Where is " + str(participants[entity_id]) + "?!</s>"
                    qa_stories = [question + story] * len(states)
#                 print(qa_stories[0])
#                 stories = [story] * len(states)
                bert_tokenizer_fast = roberta_tokenizer_fast(qa_stories, return_tensors='pt').to(device)
                bert_tokenizer_fast, padding = roberta_extract_timestamp_sequence(bert_tokenizer_fast, end_time=len(states))
                padding = padding - 1
                candidate_spans = sample['candidate_spans']
                candidate_spans = [a + b for a, b in zip(candidate_spans, [padding] * len(candidate_spans))]
                status_labels = []
                start_positions = []
                end_positions = []
                action_label = []
                for step, state in enumerate(states):
                    status_label = 0
                    if state[0] == "?":
                        status_label = 1
                        status_total["?"] += 1
                            
                    elif state[0] == "-":
                        status_label = 2
                        status_total["-"] += 1
                    else:
                        status_total["Location"] += 1
                        location_total += 1

                    status_label = torch.tensor(status_label).long().to(device)
                        
                    status_labels.append(status_label)
                    
                    if state[2] != -1 and state[3] != -1:
#                         print("in IF")
                        start_position = torch.tensor(state[2] + padding).to(device)
                        end_position = torch.tensor(state[3] + padding).to(device)
                    else:
                        start_position = torch.tensor(state[2]).to(device)
                        end_position = torch.tensor(state[3]).to(device)
                        
                    start_positions.append(start_position)
                    end_positions.append(end_position)
                    
                start_positions = torch.stack(start_positions)
                end_positions = torch.stack(end_positions)
                status_labels = torch.stack(status_labels)
#                 action_label = torch.tensor(action_label).to(device)
                outputs, qa_results, losses = model(**bert_tokenizer_fast, status_answer=status_labels, start_positions=start_positions, 
                                                        end_positions=end_positions, test=True)

                token_inputs = roberta_tokenizer_fast(qa_stories, return_tensors='pt')['input_ids'][0]
                all_tokens = roberta_tokenizer_fast.convert_ids_to_tokens(token_inputs)
                    
                outputs1 = outputs['start_logits'][:, candidate_spans]
                outputs2 = outputs['end_logits'][:, candidate_spans]
#                 outputs1 = outputs['start_logits']
#                 outputs2 = outputs['end_logits']

                max1, max_idx1 = torch.max(outputs1, -1)
                max2, max_idx2 = torch.max(outputs2, -1)
#                 print(all_tokens)
                spans = []
                for step, state in enumerate(states):
                    final_max1 = (outputs['start_logits'][step]==max1[step]).nonzero()[0][0].item()
                    final_max2 = (outputs['start_logits'][step]==max1[step]).nonzero()[0][0].item()
                    answer = ''.join(all_tokens[final_max1 : final_max2 + 1])
                    if len(answer) and answer[0] == "Ġ":
                        answer = answer[1:]
                    answer = answer.replace('Ġ', ' ')
                    if inflect.singular_noun(answer) != False:
                        answer = inflect.singular_noun(answer)                    
                    if status_labels[step].item() == 0:
                        Canswer = ' '.join(all_tokens[state[2]+padding : state[3] + padding + 1])
                        Canswer = Canswer.replace('Ġ', '')
#                         print(state[0])
                        if inflect.singular_noun(Canswer) != False:
                            Canswer = inflect.singular_noun(Canswer)
                        if state[2] == -1:
                            Canswer = state[0]
                    spans.append(answer)
                    if location_match(answer, state[0]) and status_labels[step].item() == 0:
                        location_correct += 1

                classification_answer = torch.argmax(qa_results, -1)
                qa_classifers = classification_answer.tolist()
                sample_data['entity_step'].append(torch.softmax(qa_results, -1).tolist())
#                 print(torch.softmax(qa_results, -1))
                for idx, cls_answer in enumerate(classification_answer):
                    if cls_answer.item() == status_labels[idx].item():
                        if cls_answer.item() == 0:
                            status_correct["Location"] += 1
                        elif cls_answer.item() == 1:
                            status_correct["?"] += 1
                        else:
                            status_correct["-"] += 1

                sample_output = make_eval_entry(qa_classifers, spans, participants[entity_id], sample['para_id'])
                sample_output_logical = make_eval_entry_logical(qa_classifers, spans, participants[entity_id], sample['para_id'])
                predicted_states = make_states(qa_classifers, spans=spans)
                table_entry['entities'][participants[entity_id]] = predicted_states
#                 print(sample_output)
                data.extend(sample_output)
                data_logical.extend(sample_output_logical)
            
            table[sample['para_id']] = table_entry
            for key, item in it_status_total.items():
                it_test_status_total[key] += status_total[key]
                it_test_status_correct[key] += status_correct[key]
                
#             for key,item in it_change_total.items():
#                 it_change_total[key] += change_total[key]
#                 it_change_correct[key] += change_correct[key]

            it_test_location_total += location_total
            it_test_location_correct += location_correct
            iter_data.append(sample_data)
        except KeyboardInterrupt:
            raise
        except:
            torch.cuda.empty_cache()
            print("one passed")
            raise
        
    print("The test iteration ", str(iteration), " final results are: ")
    print(it_test_location_total, it_test_location_correct)
    print(it_test_status_total, it_test_status_correct)
    print("The location accuracy is: ", it_test_location_correct / it_test_location_total)
    status_accuracy_test = (it_test_status_correct["-"] + it_test_status_correct["?"] + it_test_status_correct["Location"]) / (it_test_status_total["-"] + it_test_status_total["?"] + it_test_status_total["Location"])
    print("The status accuracy is: ", status_accuracy_test)
#     print("change: ", it_change_total, it_change_correct)
#     change_accuracy_test = (it_change_correct["NC"] + it_change_correct["M"] + it_change_correct["C"] + it_change_correct['D']) / (it_change_total["NC"] + it_change_total["M"] + it_change_total["C"] + it_change_total["D"])
#     print("The change accuracy is: ", change_accuracy_test)
    
    import csv
    csv_file = "saves/" + str(name) + "_" + str(it) + "_output.tsv"
    csv_file_logical = "saves/" + str(name) + "_" + str(it) + "_output_logical.tsv"
    try:
        with open(csv_file, 'w', newline='') as f_output:
            tsv_output = csv.writer(f_output, delimiter='\t')
            for item in data:
                item = [item['para_id'], item['step'], item['entity'], item['action'], item['before'], item['after']]
                tsv_output.writerow(item)
                
        with open(csv_file_logical, 'w', newline='') as f_output:
            tsv_output = csv.writer(f_output, delimiter='\t')
            for item in data_logical:
                item = [item['para_id'], item['step'], item['entity'], item['action'], item['before'], item['after']]
                tsv_output.writerow(item)
    except IOError:
        print("I/O error")
    
    if name=="test":
        get_metrics(test_propara_qa, table)
        
    with open("saves/" + name + "_data.json", 'w') as outfile:
        json.dump(iter_data, outfile)
    return status_accuracy_test ,(it_test_location_correct / it_test_location_total)


import matplotlib.pyplot as plt
import inflect
inflect = inflect.engine()

only_test = 'n'
while True:
    only_test = input("Run only Test (y/n):")
    if only_test == 'y' or only_test == 'n':
        break
        
if only_test == 'y':
    start_it = 300
    end_it = 301
    Test = True
    start_e = 0
    end_e = 0
else:
    start_it = 0
    end_it = 150
    Test = False
    start_e = 0
    end_e = len(train_propara_roberta_qa)

optimizer = torch.optim.SGD(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
device = 'cuda:0'
if Test:
    model.load_state_dict(torch.load("./saves/best_model"))
model.to(device)
classifier_option = 1

best_status = 0
best_location = 0
best_sum = 0
all_losses = []
dev_sum = []
for iteration in tqdm(range(start_it,end_it)):
    it_total_loss = 0
    it_location_total = 0
    it_location_correct = 0
    it_status_total = {"-": 0, "?": 0, "Location": 0}
    it_status_correct = {"-": 0, "?": 0, "Location": 0}
    it_change_total = {"NC": 0, "M": 0, "C": 0, "D": 0}
    it_change_correct = {"NC": 0, "M": 0, "C": 0, "D": 0}
#     random.shuffle(train_propara_roberta_qa)
    model.train()
    for sample_id in tqdm(range(len(train_propara_roberta_qa[start_e:end_e]))):
        try:
            sample = train_propara_roberta_qa[sample_id]
#             print(sample['para_id'])
            story = sample['sentence_paragraph']
#             print("story is : ", story)
            participants = sample['participants']
        #     print(story)
            total_loss = 0
            location_total = 0
            location_correct = 0
            status_total = {"-": 0, "?": 0, "Location": 0}
            status_correct = {"-": 0, "?": 0, "Location": 0}
            change_total = {"NC": 0, "M": 0, "C": 0, "D": 0}
            change_correct = {"NC": 0, "M": 0, "C": 0, "D": 0}
#             print(len(sample['states_annotation']))
            for entity_id, states in enumerate(sample['states_annotation']):
#                 print(states)
                question = "Where is " + str(participants[entity_id]) + "?!</s>"
                qa_stories = [question + story] * len(states)
#                 print(qa_stories[0])
#                 stories = [story] * len(states)
                bert_tokenizer_fast = roberta_tokenizer_fast(qa_stories, return_tensors='pt').to(device)
                bert_tokenizer_fast, padding = roberta_extract_timestamp_sequence(bert_tokenizer_fast, end_time=len(states))
                padding = padding - 1
                candidate_spans = sample['candidate_spans']
                candidate_spans = [a + b for a, b in zip(candidate_spans, [padding] * len(candidate_spans))]
                status_labels = []
                start_positions = []
                end_positions = []
                for step, state in enumerate(states):
                    status_label = 0
                    if state[0] == "?":
                        status_label = 1
                        status_total["?"] += 1
                            
                    elif state[0] == "-":
                        status_label = 2
                        status_total["-"] += 1
                    else:
                        status_total["Location"] += 1
                        location_total += 1

                    status_label = torch.tensor(status_label).long().to(device)
                    
                    status_labels.append(status_label)
                    
                    if state[2] != -1 and state[3] != -1:
#                         print("in IF")
                        start_position = torch.tensor(state[2] + padding).to(device)
                        end_position = torch.tensor(state[3] + padding).to(device)
                    else:
                        start_position = torch.tensor(state[2]).to(device)
                        end_position = torch.tensor(state[3]).to(device)
                        
                    start_positions.append(start_position)
                    end_positions.append(end_position)
                    
                start_positions = torch.stack(start_positions)
                end_positions = torch.stack(end_positions)
                status_labels = torch.stack(status_labels)
                outputs, qa_results, losses = model(**bert_tokenizer_fast, status_answer = status_labels, start_positions=start_positions, 
                                                        end_positions=end_positions)
                
                token_inputs = roberta_tokenizer_fast(qa_stories, return_tensors='pt')['input_ids'][0]
                all_tokens = roberta_tokenizer_fast.convert_ids_to_tokens(token_inputs)
                    
                outputs1 = outputs['start_logits'][:, candidate_spans]
                outputs2 = outputs['end_logits'][:, candidate_spans]


                max1, max_idx1 = torch.max(outputs1, -1)
                max2, max_idx2 = torch.max(outputs2, -1)
#                 print(all_tokens)
                for step, state in enumerate(states):
                    final_max1 = (outputs['start_logits'][step]==max1[step]).nonzero()[0][0].item()
                    final_max2 = (outputs['start_logits'][step]==max1[step]).nonzero()[0][0].item()
                    answer = ''.join(all_tokens[final_max1 : final_max2 + 1])
                    if len(answer) and answer[0] == "Ġ":
                        answer = answer[1:]
                    answer = answer.replace('Ġ', ' ')
                    if inflect.singular_noun(answer) != False:
                        answer = inflect.singular_noun(answer)                    
                    if status_labels[step].item() == 0:
                        Canswer = ' '.join(all_tokens[state[2]+padding : state[3] + padding + 1])
                        if len(Canswer) and Canswer[0] == "Ġ":
                            Canswer = Canswer[1:]
                        Canswer = Canswer.replace('Ġ', ' ')
#                         print(state[0])
#                         if inflect.singular_noun(Canswer) != False:
#                             Canswer = inflect.singular_noun(Canswer)
                        if state[2] == -1:
                            Canswer = state[0]
                    if location_match(answer, state[0]) and status_labels[step].item() == 0:
                        location_correct += 1

                classification_answer = torch.argmax(qa_results, -1)
                
                for idx, cls_answer in enumerate(classification_answer):
                    if cls_answer.item() == status_labels[idx].item():
                        if cls_answer.item() == 0:
                            status_correct["Location"] += 1
                        elif cls_answer.item() == 1:
                            status_correct["?"] += 1
                        else:
                            status_correct["-"] += 1


                if losses[0] is not None:
                    total_loss += losses[0]
                if losses[1] is not None:
                    total_loss += losses[1]
                    

#             print("total loss of example", sample_id," is: ", total_loss)
            if total_loss != 0 :
                total_loss = total_loss / len(participants)
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                it_total_loss += total_loss.item()
                del(total_loss)

            for key, item in it_status_total.items():
                it_status_total[key] += status_total[key]
                it_status_correct[key] += status_correct[key]
            
            for key,item in it_change_total.items():
                it_change_total[key] += change_total[key]
                it_change_correct[key] += change_correct[key]

            it_location_total += location_total
            it_location_correct += location_correct
        except KeyboardInterrupt:
            raise
        except:
            torch.cuda.empty_cache()
            print("one passed")
#             raise
    if not Test:
        scheduler.step()
        print("The iteration loss is: ", it_total_loss)
        all_losses.append(it_total_loss)
        plt.figure()
        plt.plot(all_losses, label="Loss")
        plt.legend()
        plt.savefig('saves/train_plot_loss.png')
        plt.close()
        print("The iteration ", str(iteration), " final results are: ")
        print(it_location_total, it_location_correct)
        print(it_status_total, it_status_correct)
        print("The location accuracy is: ", it_location_correct / it_location_total)
        print("The status accuracy is: ", (it_status_correct["-"] + it_status_correct["?"] + it_status_correct["Location"]) / (it_status_total["-"] + it_status_total["?"] + it_status_total["Location"]))
        torch.save(model.state_dict(), "saves/last_model")
        
    model.eval()
    status_accuracy_test, location_accuracy_test = test_model(model, dev_propara_roberta_qa, roberta_tokenizer_fast, roberta_tokenizer, name="dev", it=iteration)

    if not Test:
        dev_sum.append(status_accuracy_test + location_accuracy_test)
        plt.figure()
        plt.plot(dev_sum, label="Acc_Sum")
        plt.legend()
        plt.savefig('saves/dev_acc_sum.png')
        plt.close()

        if best_status < status_accuracy_test:
            torch.save(model.state_dict(), "saves/best_status_model")
            best_status = status_accuracy_test
        if best_location < location_accuracy_test:
            torch.save(model.state_dict(), "saves/best_location_model")
            best_location = location_accuracy_test
        if best_sum < status_accuracy_test + location_accuracy_test:
            torch.save(model.state_dict(), "saves/best_model")
            best_sum = status_accuracy_test + location_accuracy_test
        
    status_accuracy_test, location_accuracy_test = test_model(model, test_propara_roberta_qa, roberta_tokenizer_fast, roberta_tokenizer, name="test", it=iteration)
