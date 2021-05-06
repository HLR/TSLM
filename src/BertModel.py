from transformers import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from typing import Union, List
import numpy

from torch.autograd import Variable

class BertEMB(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
#         config.hidden_dropout_prob = 0.0
#         config.attention_probs_dropout_prob = 0.0
#         config.max_position_embeddings = 1024
        self.bert = BertModel(config)       
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):  
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

#         print(input_ids.shape)
        outputs = outputs[0]
        return outputs 
