import json
import spacy
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertTokenizerFast, RobertaTokenizerFast, RobertaTokenizer, AutoTokenizer
import re
import inflect

nlp = spacy.load('en_core_web_lg')

roberta_tokenizer_fast = RobertaTokenizerFast.from_pretrained('roberta-large')

engine = inflect.engine()
step_in_text = False

def update_propara_data(examples):
    propara_qa = []
    for sample_id, sample in enumerate(examples):
        sentences = ""
        for s_id, sentence in enumerate(sample['sentence_texts']):
            if s_id != len(sample['sentence_texts']) - 1:
                if not step_in_text:
                    sentences += sentence + ' </s> '
                else:
                    sentences += sentence + 'at step ' +str(s_id+1) +' </s> '
            else:
                if not step_in_text:
                    sentences += sentence 
                else:
                    sentences += sentence + 'at step ' +str(s_id+1)
#         print(sentences)
    #     sentences = sentences.replace("recyling bin", "recyle bin")
    #     sentences = sentences.replace("recyling facility", "recyle facility")
        sample['sentence_paragraph'] = sentences
        output = nlp(sentences)
#         print([token.text for token in output])
    #     value = 'SEP'
    #     for m in re.finditer(value.lower(), sentences.lower()):
    #         print(value, ' found', m.start(), m.end())
        bert_tokenizer_fast = roberta_tokenizer_fast(sentences, return_offsets_mapping=True, return_tensors='pt')
#         all_tokens = roberta_tokenizer_fast.convert_ids_to_tokens(bert_tokenizer_fast['input_ids'][0])
#         print(all_tokens)
#         print(bert_tokenizer_fast)
        token_starts = [-1]
        for token in bert_tokenizer_fast['offset_mapping'][0][1:-1]:
            token_starts.append(token[0].item())
        token_starts.append(-1)

        token_ends = [-1]
        for token in bert_tokenizer_fast['offset_mapping'][0][1:-1]:
            token_ends.append(token[1].item())
        token_ends.append(-1)

        noun_list = ["NN", "PROPN", "NNP", "NNS", "NNPS", "NOUN", "ADV", "ADJ", "ADP", "DET", "SCONJ"]
        spacy_starts = []
        spacy_ends = []
        final_ids = []
        for token in output:
#             print(token.text)
            if token.pos_ in noun_list and token.text != "/s":
                spacy_starts.append(token.idx)
                spacy_ends.append(token.idx + len(token.text))
                if token.idx in token_starts:
                    if token_starts.index(token.idx) not in final_ids:
                        final_ids.append(token_starts.index(token.idx))
                else:
                    print("error 1")

                if token.idx + len(token.text) in token_ends:
                    if token_ends.index(token.idx + len(token.text)) not in final_ids:
                        final_ids.append(token_ends.index(token.idx + len(token.text)))
                else:
                    print("error 2")
        sample['candidate_spans'] = final_ids

        boundaries = []
        start = 0
        for m in re.finditer('/s'.lower(), sentences.lower()):
            if sentences[m.end()] == ">":
                boundaries.append((start, m.start()-2))
                start = m.end() + 2
    #         print(' found', m.start(), m.end())
        boundaries.append((start, len(sentences)))
    #     print(boundaries)
        sample['boundaries'] = boundaries
        sample['states_annotation'] = []
        for entity_num, state in enumerate(sample['states']):
            sample['states_annotation'].append([])
            prev_loc = ""
            for time, loc in enumerate(state):
                if "'" in loc:
    #                 print(loc)
                    loc = loc.replace(" '", "'")
                all_loc = []
                final_loc = (0, 0)
                if loc == "nil":
                    loc = "-"
                if loc != "-" and loc != "?":
                    if loc == prev_loc:
                        final_loc = sample['states_annotation'][-1][-1][1]
                        bert_start_token = sample['states_annotation'][-1][-1][2]
                        bert_end_token = sample['states_annotation'][-1][-1][3]
                        sample['states_annotation'][-1].append((loc, final_loc, bert_start_token, bert_end_token))
                        prev_loc = loc
                        continue
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
    #                 if len(all_loc) == 0:
    #                     final_loc = final_loc
                    if len(all_loc) == 0 and "recycle" in loc:
                        for m in re.finditer(" " + loc.replace("recycle", "recycling").lower(), sentences.lower()):
                            start = m.start()
                            if sentences[m.start()] == " ":
                                start = m.start() + 1
                            all_loc.append((start, m.end()))
                    if len(all_loc) == 0:
                        if loc == "alveolus":
                            loc = "alveoli"
                        if loc == "sew machine":
                            loc = "machine"
                        if loc == "cool tower":
                            loc = "cooling tower"
                        if loc == "cart or on a conveyor belt":
                            loc = "carts or on a conveyor belt"
                        if loc == "bee leg":
                            loc = "bees legs"
                        if loc == "bottom of river and ocean":
                            loc = "bottom of rivers and oceans"
                        if loc == "body of water":
                            loc = "bodies of water"
                        if loc == "crack in rock":
                            loc = "cracks in rocks"
                        if loc == "dry ingredient .":
                            loc = "dry ingredients"
                        if loc == "grease cake pan":
                            loc = "greased cake pan"
                        if loc == "release from the atom":
                            loc = "released from the atom"
                        if loc == "bottom of ocean , riverbed or swamp":
                            loc = "bottom of oceans, riverbeds or swamps"
                        if loc == "opposite end of the cell" or loc == "opposite pole of the cell":
                            loc = "opposite poles of the cell"
                        if loc == "fat , muscle and liver cell":
                            loc = "fat, muscle and liver cells"
                        if loc == "turn mechanisms":
                            loc = "turning mechanism"
                        if loc == "surround rocks":
                            loc = "sorrounding rocks"
                        for m in re.finditer(" " + loc.lower(), sentences.lower()):
                            start = m.start()
                            if sentences[m.start()] == " ":
                                start = m.start() + 1
                            all_loc.append((start, m.end()))
                    if len(all_loc) == 0:
                        loc = loc.replace(" , ", ", ")
                        for m in re.finditer(" " + loc.lower(), sentences.lower()):
                            start = m.start()
                            if sentences[m.start()] == " ":
                                start = m.start() + 1
                            all_loc.append((start, m.end()))
                    if len(all_loc) == 0:
                        loc = loc.replace(" , ", ", ")
                        stri = loc.split(",")
                        stri_f = ""
                        for item in stri:
                            if not engine.singular_noun(item):
                                item = engine.plural(item)
                            stri_f += "," + item
                        loc = stri_f[1:]
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
                        if len(all_loc) == 0:
                            stri = loc.split("and")
                            stri_f = ""
                            for item in stri:
                                if not engine.singular_noun(item):
                                    item = engine.plural(item)
                                stri_f += "and" + item
                            loc = stri_f[3:]
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
                        if len(all_loc) == 0:
                            print("data in hand 3: ", loc)
#                             print("story: ", sentences)
#                             raise ValueError("item not found anywhere")
                    if len(all_loc) == 1 or (not time and len(all_loc) >= 1):
                        final_loc = all_loc[0]
                    else:
                        in_sentence_check = False
    #                     try:
                        if time:
                            for can_loc in all_loc:
                                if can_loc[0] >= boundaries[time-1][0] and can_loc[1] <= boundaries[time-1][1]:
                                    final_loc = can_loc
                                    in_sentence_check = True
                                    break
                            if not in_sentence_check:
                                if len(all_loc) == 0:
                                    selected_boundary = (0, 0)
                                else:
                                    selected_boundary = (0, 0)
                                    # check whether it is in the next sentence
                                    if len(boundaries) > time:
                                        for can_loc in all_loc:
                                            if can_loc[0] >= boundaries[time][0] and can_loc[1] <= boundaries[time][1]:
                                                selected_boundary = can_loc
                                                break
                                    if selected_boundary != (0, 0):
                                        # check whether it is before this step
                                        for can_loc in all_loc:
        #                                     print(can_loc)
                                            if can_loc[0] < boundaries[time-1][0] and can_loc[0] > selected_boundary[0]:
                                                selected_boundary = can_loc
                                        # check whether it is after this step
                                        if selected_boundary == (0,0):
                                            selected_boundary = all_loc[-1]
                                            for can_loc in all_loc: 
                                                 if can_loc[1] > boundaries[time-1][1] and can_loc[1] < selected_boundary[1]:
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
#                     print(bert_tokenizer_fast)
                    print("data in hand: ", loc, all_loc, final_loc, bert_start_token, bert_end_token)
                    print("story: ", sentences)
                    print("in sentence check: ", in_sentence_check)
                    print("step: ", time)
                    print("For entity: ", sample['participants'][entity_num])
                    print("sample_id: ", sample_id)
                    print(final_loc[0], final_loc[1])
                    print(token_ends)
                    print([(token.pos_, token.text) for token in output])
                    all_tokens = roberta_tokenizer_fast.convert_ids_to_tokens(bert_tokenizer_fast['input_ids'][0])
                    print(all_tokens)
                    raise
                sample['states_annotation'][-1].append((loc, final_loc, bert_start_token, bert_end_token))
                prev_loc = loc
        propara_qa.append(sample)
    return propara_qa