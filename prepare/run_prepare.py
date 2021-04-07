import json
from roberta_prepare import update_propara_data

def read_propara_data(file):
    with open(file, 'r') as f:
            lines = []
            for line in f:
#                 print(line)
                try:
                    if line != "\n":
                        lines.append(json.loads(str(line)))
                except:
#                     print(line)
                    raise
    return lines

train_propara = read_propara_data('data/grids.v1.train.json')
test_propara = read_propara_data('data/grids.v1.test.json')
dev_propara = read_propara_data('data/grids.v1.dev.json')
    
    
train_propara_roberta_qa = update_propara_data(train_propara)
with open('data/train_propara_roberta_version.json', 'w') as outfile:
    json.dump(train_propara_roberta_qa, outfile)
    
test_propara_roberta_qa = update_propara_data(test_propara)
with open('data/test_propara_roberta_version.json', 'w') as outfile:
    json.dump(test_propara_roberta_qa, outfile)
    
dev_propara_roberta_qa = update_propara_data(dev_propara)
with open('data/dev_propara_roberta_version.json', 'w') as outfile:
    json.dump(dev_propara_roberta_qa, outfile)