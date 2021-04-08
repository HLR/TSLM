import json, os
import sys
sys.path.append("..")

with open('cooking_dataset/npn_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    
train_set = [item for item in data if item['split'] == "train"]
dev_set = [item for item in data if item['split'] == "dev"]
test_set = [item for item in data if item['split'] == "test"]
print("tran size: ", len(train_set))
print("dev size: ", len(dev_set))
print("test size: ", len(test_set))


def get_location_samples(loop_set):
    location_examples = []
    errors = [0, 0]
    num_examples = 0
    for ind, item in enumerate(loop_set):
        generated = 0
        try:
            for step, event in item['events'].items():
#                 print(step, event)
                if "location" in event:
                    sample = {}
                    sample['text'] = item['text']
#                 [" ".join(text) for tid, text in item['text'].items()]
#                     sample['step_text'] = item['text'][step]
                    sample['step'] = step
                    sample['ingredients'] = [item['ingredient_list'][ing] for ing in item['ingredients'][step]]
                    sample['all_ings'] = item['ingredient_list']
                    sample['event'] = event
                    location_examples.append(sample)
                    generated = 1
        except:
            if step == "0":
                errors[0] += 1
            else:
                errors[1] += 1
    #             print(item)
    #             print(item['ingredients'])
    #             print(event)
    #             print(step)
    #             print(item['ingredient_list'])
    #             print(item['ingredients'][step])
    #             raise

    #             print(ind, step, event, [item['ingredient_list'][ing] for ing in item['ingredients'][step]], item['text'][step])
        num_examples += generated
    return location_examples, errors, num_examples


train_locations, ter, tgen = get_location_samples(train_set)
dev_locations, der, dgen = get_location_samples(dev_set)
test_locations, teer, tegen = get_location_samples(test_set)
print("number of location instances: ", len(train_locations), len(dev_locations), len(test_locations))
print("errors: ", ter, der, teer)
print("sample number: ", tgen, dgen, tegen)


import torch
from transformers import RobertaTokenizerFast, RobertaTokenizer
from tqdm import tqdm
# roberta_tokenizer_fast = RobertaTokenizerFast.from_pretrained('phiyodr/roberta-large-finetuned-squad2')
roberta_tokenizer_fast = RobertaTokenizerFast.from_pretrained('roberta-large')

import re
import inflect
engine = inflect.engine()

def roberta_update_npn_data(examples):
    ncn_qa = []
    for sample_id in tqdm(range(len(examples))):
        sample = examples[sample_id]
        sentences = ""
        sample['fixed_text'] = ["" for dtext in sample['text']]
        for idt, dtext in sample['text'].items():
            sample['fixed_text'][int(idt)] = dtext
        sample['fixed_text'] = [" ".join(sentence) for sentence in sample['fixed_text']]
        for s_id, sentence in enumerate(sample['fixed_text']):
            if s_id != len(sample['fixed_text']) - 1:
                sentences += sentence + ' </s> '
            else:
                sentences += sentence

        sample['sentence_paragraph'] = sentences
        
#         print(sentences)

        bert_tokenizer_fast = roberta_tokenizer_fast(sentences, return_offsets_mapping=True, return_tensors='pt')

        token_starts = [-1]
        for token in bert_tokenizer_fast['offset_mapping'][0][1:-1]:
            token_starts.append(token[0].item())
        token_starts.append(-1)

        token_ends = [-1]
        for token in bert_tokenizer_fast['offset_mapping'][0][1:-1]:
            token_ends.append(token[1].item())
        token_ends.append(-1)

        boundaries = []
        start = 0
        for m in re.finditer('/s'.lower(), sentences.lower()):
            boundaries.append((start, m.start()-2))
            start = m.end() + 2
    #         print(' found', m.start(), m.end())
        boundaries.append((start, len(sentences)))
    #     print(boundaries)
        sample['boundaries'] = boundaries
        sample['annotation'] = []
        sample['not_in_text'] = 0
        for entity in sample['ingredients']:
#             print(entity)
            loc = sample['event']['location'][0]
            time = int(sample['step'])
#             print(loc, time)
            all_loc = []
            final_loc = (0, 0)
            for m in re.finditer(" " + loc.lower(), sentences.lower()):
                start = m.start()
                if sentences[m.start()] == " ":
                    start = m.start() + 1
                all_loc.append((start, m.end()))
            
            if len(all_loc) == 0:
                if loc == "fridge":
                    loc = "refrigerate"
                    for m in re.finditer(" " + loc.lower(), sentences.lower()):
                        start = m.start()
                        if sentences[m.start()] == " ":
                            start = m.start() + 1
                        all_loc.append((start, m.end()))
            if len(all_loc) == 0:
                for m in re.finditer(loc.lower(), sentences.lower()):
                    start = m.start()
                    if sentences[m.start()] == " ":
                        start = m.start() + 1
                    all_loc.append((start, m.end()))

            if len(all_loc) == 1 or (not time and len(all_loc) >= 1):
                final_loc = all_loc[0]
            else:
                in_sentence_check = False
#                     try:
                if time:
                    for can_loc in all_loc:
                        if can_loc[0] > boundaries[time][0] and can_loc[1] < boundaries[time][1]:
                            final_loc = can_loc
                            in_sentence_check = True
                            break
                    if not in_sentence_check:
                        if len(all_loc) == 0:
                            selected_boundary = (0, 0)
                        else:
                            selected_boundary = (0, 0)
                            for can_loc in all_loc:
#                                     print(can_loc)
                                if can_loc[0] < boundaries[time][0] and can_loc[0] > selected_boundary[0]:
                                    selected_boundary = can_loc
                            if selected_boundary == (0,0):
                                selected_boundary = all_loc[-1]
                                for can_loc in all_loc: 
                                     if can_loc[1] > boundaries[time][1] and can_loc[1] < selected_boundary[1]:
                                            selected_boundary = can_loc
                        final_loc = selected_boundary

    #                     except:
    #                         print(time)
    #                         print(can_loc)
    #                         raise

    #             print(loc, all_loc, final_loc)
            bert_start_token = -1
            bert_end_token = -1
            try:
                if final_loc[0] != 0 or final_loc[1] != 0:
                    if final_loc[0] in token_starts:
                        bert_start_token = token_starts.index(final_loc[0])
                    elif final_loc[0]-1 in token_starts:
                        bert_start_token = token_starts.index(final_loc[0]-1)
                    else:
                        bert_start_token = token_starts.index(final_loc[0])
#                     print(bert_start_token)
                    if token_ends[bert_start_token] > final_loc[1]:
                        bert_end_token = bert_start_token
                    else:
                        if final_loc[1] in token_ends:
                            bert_end_token = token_ends.index(final_loc[1])
                        elif final_loc[1] + 1 in token_ends:
                            bert_end_token = token_ends.index(final_loc[1] + 1)
                        elif final_loc[1] + 2 in token_ends:
                            bert_end_token = token_ends.index(final_loc[1] + 2)
                        elif final_loc[1] + 3 in token_ends:
                            bert_end_token = token_ends.index(final_loc[1] + 3)
                        else:
                            raise ValueError("the bert end not found")

#                         if bert_start_token not in final_ids:
#                             raise ValueError("the value is not a candidate")
#                         print(bert_end_token)
            except:
                print(bert_tokenizer_fast)
                print("data in hand: ", loc, all_loc, final_loc, bert_start_token, bert_end_token)
                print(sample['event']['location'])
                print("story: ", sentences)
                print("in sentence check: ", in_sentence_check)
                print("step: ", time)
#                 print("For entity: ", sample['participants'][entity_num])
#                 print("sample_id: ", sample_id)
                print(final_loc[0], final_loc[1])
                print(token_ends)
                all_tokens = roberta_tokenizer_fast.convert_ids_to_tokens(bert_tokenizer_fast['input_ids'][0])
                print(all_tokens)
                sample['not_in_text'] = 1
#                 raise
            sample['annotation'].append((loc, final_loc, bert_start_token, bert_end_token))
        
        sample["extra_ings"] = []
#         stop_point = len(sample['ingredients'])
#         for extra_ing in sample['all_ings']:
#             if extra_ing not in sample['ingredients']:
#                 sample['annotation'].append(("-", (-1, -1), -1, -1))
#                 sample["extra_ings"].append(extra_ing)
#                 stop_point -= 1
#             if stop_point == -1:
#                 break
#             print(sample['annotation'])
        ncn_qa.append(sample)
    return ncn_qa
    
    
npn_qa_dev = roberta_update_npn_data(dev_locations)
npn_qa_test = roberta_update_npn_data(test_locations)
npn_qa_train = roberta_update_npn_data(train_locations[0:15000])


def roberta_extract_timestamp_sequence(inputs, end_time):
    f_out = []
    padding = []
    time = end_time
    for idx in range(len(inputs['input_ids'])):
        timestamp_id = []
        check = -1
        for index, ids in enumerate(inputs['input_ids'][idx]):
            if ids == 2:
                check += 1
                if check == 0:
                    padding.append(index + 1)
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

from roberta import RobertaProceduralNPNQA

model = RobertaProceduralNPNQA.from_pretrained('tli8hf/unqover-roberta-large-squad', return_dict=True)

def test_npn(model, samples, roberta_tokenizer_fast, name):
    it_location_total = 0
    it_location_correct = 0
    it_status_total = {"-": 0, "Location": 0}
    it_status_correct = {"-": 0, "Location": 0}
    for sample_id in tqdm(range(len(samples))):
        try:
            sample = samples[sample_id]
#             print(sample['para_id'])
            story = sample['sentence_paragraph']
#             print("story is : ", story)
            participants = sample['ingredients']
            participants.extend(sample['extra_ings'])
        #     print(story)
            total_loss = 0
            location_total = 0
            location_correct = 0
            status_total = {"-": 0, "Location": 0}
            status_correct = {"-": 0, "Location": 0}
#             print(len(sample['states_annotation']))
            status_labels = []
            qa_stories = []
            for entity_id, states in enumerate(sample['annotation']):
#                 print(sample['ingredients'][entity_id], states)
                question = "Where is " + str(participants[entity_id]) + "?!</s>"
                qa_stories.append(question + story)
                if states[0] != "-":
                    location_total += 1
                    status_labels.append(0)
                    status_total["Location"] += 1
                else:
                    status_labels.append(1)
                    status_total["-"] += 1
#                 print(qa_stories[0])
#                 stories = [story] * len(states)
                
            bert_tokenizer_fast = roberta_tokenizer_fast(qa_stories, return_tensors='pt', padding=True).to(device)
            bert_tokenizer_fast, padding = roberta_extract_timestamp_sequence(bert_tokenizer_fast, end_time=int(sample['step']))
#             print(padding)
#             print(bert_tokenizer_fast)
#                 print(sample['step'])
            padding = [pad - 1 for pad in padding] 
                
                
                    
            status_labels = torch.tensor(status_labels).long().to(device)
            start_positions = []
            end_positions = []
            for entity_id, states in enumerate(sample['annotation']):
                if states[2] != -1 and states[3] != -1:
#                         print("in IF")
                    start_positions.append(torch.tensor(states[2] + padding[entity_id]).to(device))
                    end_positions.append(torch.tensor(states[3] + padding[entity_id]).to(device))
                else:
                    start_positions.append(torch.tensor(states[2]).to(device))
                    end_positions.append(torch.tensor(states[3]).to(device))
                    
            start_positions = torch.stack(start_positions)
            end_positions = torch.stack(end_positions)
            outputs, loss = model(**bert_tokenizer_fast, status_answer = status_labels, start_positions=start_positions, 
                                                    end_positions=end_positions, test=True)
                
            token_inputs = roberta_tokenizer_fast(qa_stories, return_tensors='pt', padding=True)['input_ids']

            outputs1 = outputs['start_logits']
            outputs2 = outputs['end_logits']

            max1, max_idx1 = torch.max(outputs1, -1)
            max2, max_idx2 = torch.max(outputs2, -1)
            for entity_id, states in enumerate(sample['annotation']):
#                 print("entity: ", participants[entity_id], padding[entity_id])
                all_tokens = roberta_tokenizer_fast.convert_ids_to_tokens(token_inputs[entity_id])
#                 print(all_tokens, states)
                answer = ' '.join(all_tokens[max_idx1[entity_id] : max_idx2[entity_id] + 1])
                answer2 = ''.join(all_tokens[max_idx1[entity_id] : max_idx2[entity_id] + 1])
                answer = answer.replace('Ġ', '')
                answer2 = answer2.replace('Ġ', '')
#                 print("prediction: ", answer, answer2)
#                 print("annotation: ", states[0], sample['event']['location'])
                if status_labels[entity_id].item() == 0:
                    Canswer = ' '.join(all_tokens[states[2]+padding[entity_id] : states[3] + padding[entity_id] + 1])
                    Canswer = Canswer.replace('Ġ', '')
                    if states[2] == -1:
                        Canswer = states[0]
#                     print("the supervision: ", Canswer)
                if (location_match(answer, states[0]) or location_match(answer2, states[0]))and status_labels[entity_id].item() == 0:
                    location_correct += 1

            it_location_total += location_total
            it_location_correct += location_correct
        except KeyboardInterrupt:
            raise
        except:
            torch.cuda.empty_cache()
            print("one passed")
#             raise
    
    print("Test final results are: ")
    print(it_location_total, it_location_correct)
    print("The location accuracy is: ", it_location_correct / it_location_total)
    return it_location_correct / it_location_total


import matplotlib.pyplot as plt
import inflect
import random
inflect = inflect.engine()

only_test = 'n'
while True:
    only_test = input("Run only Test (y/n):")
    if only_test == 'y' or only_test == 'n':
        break
        
if only_test == 'y':
    start_it = 500
    end_it = 501
    Test = True
    start_e = 0
    end_e = 0
else:
    start_it = 0
    end_it = 100
    Test = False
    start_e = 0
    end_e = 10
#     end_e = len(npn_qa_train)
    
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
device = 'cuda:0'
# model.load_state_dict(torch.load("npn_saves/best_location_model"))
model.to(device)
classifier_option = 1

best_status = 0
best_location = 0
best_sum = 0
all_losses = []
dev_sum = []
random.shuffle(npn_qa_train)
for iteration in tqdm(range(start_it,end_it)):
    it_total_loss = 0
    it_location_total = 0
    it_location_correct = 0
    it_status_total = {"-": 0, "Location": 0}
    it_status_correct = {"-": 0, "Location": 0}
#     random.shuffle(npn_qa_train)
    model.train()
    for sample_id in tqdm(range(len(npn_qa_train[start_e:end_e]))):
        try:
            sample = npn_qa_train[sample_id]
#             print(sample['para_id'])
            story = sample['sentence_paragraph']
#             print("story is : ", story)
            participants = sample['ingredients']
            participants.extend(sample['extra_ings'])
        #     print(story)
            total_loss = 0
            location_total = 0
            location_correct = 0
            status_total = {"-": 0, "Location": 0}
            status_correct = {"-": 0, "Location": 0}
#             print(len(sample['states_annotation']))
            status_labels = []
            qa_stories = []
            for entity_id, states in enumerate(sample['annotation']):
#                 print(sample['ingredients'][entity_id], states)
                question = "Where is " + str(participants[entity_id]) + "?!</s>"
                qa_stories.append(question + story)
                if states[0] != "-":
                    location_total += 1
                    status_labels.append(0)
                    status_total["Location"] += 1
                else:
                    status_labels.append(1)
                    status_total["-"] += 1
#                 print(qa_stories[0])
#                 stories = [story] * len(states)
                
            bert_tokenizer_fast = roberta_tokenizer_fast(qa_stories, return_tensors='pt', padding=True).to(device)
            bert_tokenizer_fast, padding = roberta_extract_timestamp_sequence(bert_tokenizer_fast, end_time=int(sample['step']))
#             print(padding)
#             print(bert_tokenizer_fast)
#                 print(sample['step'])
            padding = [pad - 1 for pad in padding] 
                
                
                    
            status_labels = torch.tensor(status_labels).long().to(device)
            start_positions = []
            end_positions = []
            for entity_id, states in enumerate(sample['annotation']):
                if states[2] != -1 and states[3] != -1:
#                         print("in IF")
                    start_positions.append(torch.tensor(states[2] + padding[entity_id]).to(device))
                    end_positions.append(torch.tensor(states[3] + padding[entity_id]).to(device))
                else:
                    start_positions.append(torch.tensor(states[2]).to(device))
                    end_positions.append(torch.tensor(states[3]).to(device))
                    
            start_positions = torch.stack(start_positions)
            end_positions = torch.stack(end_positions)
            outputs, loss = model(**bert_tokenizer_fast, status_answer = status_labels, start_positions=start_positions, 
                                                    end_positions=end_positions)
                
            token_inputs = roberta_tokenizer_fast(qa_stories, return_tensors='pt', padding=True)['input_ids']

            outputs1 = outputs['start_logits']
            outputs2 = outputs['end_logits']

            max1, max_idx1 = torch.max(outputs1, -1)
            max2, max_idx2 = torch.max(outputs2, -1)
            for entity_id, states in enumerate(sample['annotation']):
                all_tokens = roberta_tokenizer_fast.convert_ids_to_tokens(token_inputs[entity_id])
                answer = ' '.join(all_tokens[max_idx1[entity_id] : max_idx2[entity_id] + 1])
                answer2 = ''.join(all_tokens[max_idx1[entity_id] : max_idx2[entity_id] + 1])
                answer = answer.replace('Ġ', '')
                answer2 = answer2.replace('Ġ', '')
                if status_labels[entity_id].item() == 0:
                    Canswer = ' '.join(all_tokens[states[2]+padding[entity_id] : states[3] + padding[entity_id] + 1])
                    Canswer = Canswer.replace('Ġ', '')
                    if states[2] == -1:
                        Canswer = states[0]
#                     print("the supervision: ", Canswer)
                if (location_match(answer, states[0]) or location_match(answer2, states[0]))and status_labels[entity_id].item() == 0:
                    location_correct += 1


            if loss is not None:
                total_loss += loss
                    

#             print("total loss of example", sample_id," is: ", total_loss)
            del(bert_tokenizer_fast, token_inputs)
            if total_loss != 0 :
#                 total_loss = (total_loss + prev_loss) / 2
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                it_total_loss += total_loss.item()
                del(total_loss)

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
        plt.savefig('npn_saves/train_plot_loss.png')
        plt.close()
        print("The iteration ", str(iteration), " final results are: ")
        print(it_location_total, it_location_correct)
        print("The location accuracy is: ", it_location_correct / it_location_total)
        torch.save(model.state_dict(), "npn_saves/last_model")
    
    model.eval()
    location_accuracy_test = test_npn(model, npn_qa_dev, roberta_tokenizer_fast, name="dev")
    
    if not Test:
        dev_sum.append(location_accuracy_test)
        plt.figure()
        plt.plot(dev_sum, label="Acc_Sum")
        plt.legend()
        plt.savefig('npn_saves/dev_acc_sum.png')
        plt.close()
        if best_location < location_accuracy_test:
            torch.save(model.state_dict(), "npn_saves/best_location_model")
            best_location = location_accuracy_test
        
    location_accuracy_test = test_npn(model, npn_qa_test, roberta_tokenizer_fast, name="test")
