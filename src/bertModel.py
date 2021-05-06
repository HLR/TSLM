# coding=utf-8
# Copyright 2019 project LXRT.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LXRT model."""

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)

BertLayerNorm = torch.nn.LayerNorm



class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=3072,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.
        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertAttention(nn.Module):
    def __init__(self, num_attention_heads=8, hidden_size=3072, drop=0.0,ctx_dim=None):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim = hidden_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(drop)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttOutput(nn.Module):
    def __init__(self, hidden_size=3072, drop=0.0):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-8)
        self.dropout = nn.Dropout(drop)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossattLayer(nn.Module):
    def __init__(self, num_attention_heads=8, hidden_size=3072, drop=0.0,ctx_dim=None):
        super().__init__()
        self.att = BertAttention(num_attention_heads=num_attention_heads, hidden_size=hidden_size, drop=drop)
        self.output = BertAttOutput(hidden_size=hidden_size, drop=drop)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output


class BertSelfattLayer(nn.Module):
    def __init__(self, num_attention_heads=8, hidden_size=3072, drop=0.0,ctx_dim=None):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(num_attention_heads=num_attention_heads, hidden_size=hidden_size, drop=drop)
        self.output = BertAttOutput(hidden_size=hidden_size, drop=drop)

    def forward(self, input_tensor, attention_mask):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size=3072, intermediate_size=2048):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = GeLU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, hidden_size=3072, intermediate_size=2048, drop=0.0):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-8)
        self.dropout = nn.Dropout(drop)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         hidden_states = hidden_states + input_tensor
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, num_attention_heads=2, hidden_size=3072, drop=0.0, intermediate_size=2048):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(num_attention_heads=num_attention_heads, hidden_size=hidden_size, drop=drop)
        self.intermediate = BertIntermediate(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.output = BertOutput(hidden_size=hidden_size, intermediate_size=intermediate_size, drop=drop)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


"""
---------------------------------------------------------------------------------------
      Above modules are copied from BERT (pytorch-transformer) with modifications.
---------------------------------------------------------------------------------------
"""


class LXRTXLayer(nn.Module):
    def __init__(self, num_attention_heads=8, hidden_size=3072, drop=0.0, intermediate_size=2048):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = BertCrossattLayer(num_attention_heads=num_attention_heads, hidden_size=hidden_size, drop=drop)

        # Self-attention Layers
        self.lang_self_att = BertSelfattLayer(num_attention_heads=num_attention_heads, hidden_size=hidden_size, drop=drop)
        self.visn_self_att = BertSelfattLayer(num_attention_heads=num_attention_heads, hidden_size=hidden_size, drop=drop)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = BertIntermediate(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.lang_output = BertOutput(hidden_size=hidden_size, intermediate_size=intermediate_size, drop=drop)
        self.visn_inter = BertIntermediate(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.visn_output = BertOutput(hidden_size=hidden_size, intermediate_size=intermediate_size, drop=drop)

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Cross Attention
        lang_att_output = self.visual_attention(lang_input, visn_input, ctx_att_mask=visn_attention_mask)
        visn_att_output = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask)
        return lang_att_output, visn_att_output

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask,
                      visn_feats, visn_attention_mask):
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        lang_att_output, visn_att_output = self.cross_att(lang_att_output, lang_attention_mask,
                                                          visn_att_output, visn_attention_mask)
        lang_att_output, visn_att_output = self.self_att(lang_att_output, lang_attention_mask,
                                                         visn_att_output, visn_attention_mask)
        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output)

        return lang_output, visn_output


class VisualFeatEncoder(nn.Module):
    def __init__(self, visual_feat_dim, visual_pos_dim, hidden_size, drop=0.0):
        super().__init__()
        feat_dim = visual_feat_dim
        pos_dim = visual_pos_dim

        # Object feature encoding
        self.visn_fc = nn.Linear(feat_dim, hidden_size)
        self.visn_layer_norm = BertLayerNorm(hidden_size, eps=1e-9)

        # Box position encoding
        self.box_fc = nn.Linear(pos_dim, hidden_size)
        self.box_layer_norm = BertLayerNorm(hidden_size, eps=1e-9)

        self.dropout = nn.Dropout(drop)

    def forward(self, vis_input):
        feats, boxes = vis_input
        
        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)
        y = self.box_fc(boxes)
        y = self.box_layer_norm(y)
        output = (x + y) / 2

        output = self.dropout(output)
        return output


class LXRTEncoder(nn.Module):
    def __init__(self, visual_feat_dim=2048, visual_pos_dim=8, drop=0.0, l_layers=9, x_layers=5, r_layers=5, num_attention_heads=4, hidden_size=3072,intermediate_size=2048):
        super().__init__()

        # Obj-level image embedding layer
        self.visn_fc = VisualFeatEncoder(visual_feat_dim=visual_feat_dim, visual_pos_dim=visual_pos_dim, hidden_size=hidden_size, drop=0.0)

        # Number of layers
        self.num_l_layers = l_layers
        self.num_x_layers = x_layers
        self.num_r_layers = r_layers
        print("LXRT encoder with %d l_layers, %d x_layers, and %d r_layers." %
              (self.num_l_layers, self.num_x_layers, self.num_r_layers))

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.layer = nn.ModuleList(
            [BertLayer(num_attention_heads=num_attention_heads, hidden_size=hidden_size, drop=drop, intermediate_size=intermediate_size) for _ in range(self.num_l_layers)]
        )
        self.x_layers = nn.ModuleList(
            [LXRTXLayer(num_attention_heads=num_attention_heads, hidden_size=hidden_size, drop=drop, intermediate_size=intermediate_size) for _ in range(self.num_x_layers)]
        )
        self.r_layers = nn.ModuleList(
            [BertLayer(num_attention_heads=num_attention_heads, hidden_size=hidden_size, drop=drop, intermediate_size=intermediate_size) for _ in range(self.num_r_layers)]
        )
        
        self.cuda_list = []

        
    def transform_cuda(self, cuda_list):
        
        self.cuda_list = cuda_list
        
        self.visn_fc.to(cuda_list[0])
        # Run language layers
        for layer_module in range(len(self.layer)):
            if layer_module < len(self.layer) / 2:
                layer_module = self.layer[layer_module]
                layer_module.to(cuda_list[1])
            else:
                layer_module = self.layer[layer_module]
                layer_module.to(cuda_list[2])

        # Run relational layers
        for layer_module in range(len(self.r_layers)):
            layer_module = self.r_layers[layer_module]
            layer_module.to(cuda_list[0])

        # Run cross-modality layers
        for layer_module in range(len(self.x_layers)):
            if layer_module < len(self.x_layers) / 2:
                layer_module = self.x_layers[layer_module]
                layer_module.to(cuda_list[3])
            else:
                layer_module = self.x_layers[layer_module]
                layer_module.to(cuda_list[2])
        
        
    def forward(self, lang_feats, lang_attention_mask,
                visn_feats, visn_attention_mask=None, logger=False):
        # Run visual embedding layer
        # Note: Word embedding layer was executed outside this module.
        #       Keep this design to allow loading BERT weights.
        visn_feats = self.visn_fc(visn_feats)

        # Run language layers
        count = 0
        check = False
        for layer_module in self.layer:
            if count < len(self.layer) / 2:
                lang_feats = layer_module(lang_feats, lang_attention_mask)
            else:
                if not check:
                    lang_feats = lang_feats.to(self.cuda_list[2])
                    check = True
                lang_feats = layer_module(lang_feats, lang_attention_mask)
            count += 1
        if logger:
            print("the language info output of self-attention shape is: ", lang_feats.shape, file=logger)
            print("the language info output of self-attention is: ", lang_feats, file=logger)
        
        # Run relational layers
        for layer_module in self.r_layers:
            visn_feats = layer_module(visn_feats, visn_attention_mask)
        
        if logger:
            print("the Vision info output of self-attention shape is: ", visn_feats.shape, file=logger)
            print("the Vision info output of self-attention is: ", visn_feats, file=logger)
        
        lang_feats = lang_feats.to(self.cuda_list[3])
        visn_feats = visn_feats.to(self.cuda_list[3])
        # Run cross-modality layers
        count = 0
        check = False
        for layer_module in self.x_layers:
            if count < len(self.x_layers) / 2:
                lang_feats, visn_feats = layer_module(lang_feats, lang_attention_mask,
                                                  visn_feats, visn_attention_mask)
            else:
                if not check:
                    lang_feats = lang_feats.to(self.cuda_list[2])
                    visn_feats = visn_feats.to(self.cuda_list[2])
                    check = True
                lang_feats, visn_feats = layer_module(lang_feats, lang_attention_mask,
                                                  visn_feats, visn_attention_mask)
            count += 1
            
        if logger:
            print("the Lang info output of cross_attention is: ", lang_feats, file=logger)    
        return lang_feats, visn_feats


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertVisualAnswerHead(nn.Module):
    def __init__(self, config, num_answers):
        super().__init__()
        hid_dim = config.hidden_size
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)




class LXRTModel(nn.Module):
    """LXRT Model."""

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = LXRTEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                visual_feats=None, visual_attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Process the visual attention mask
        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_visual_attention_mask = extended_visual_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * -10000.0
        else:
            extended_visual_attention_mask = None

        # Positional Word Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # Run LXRT backbone
        lang_feats, visn_feats = self.encoder(
            embedding_output,
            extended_attention_mask,
            visn_feats=visual_feats,
            visn_attention_mask=extended_visual_attention_mask)
        pooled_output = self.pooler(lang_feats)

        return (lang_feats, visn_feats), pooled_output

    
    
class NoPosVisualFeatEncoder(nn.Module):
    def __init__(self, visual_feat_dim, hidden_size, drop=0.0):
        super().__init__()
        feat_dim = visual_feat_dim

        # Object feature encoding
        self.visn_fc = nn.Linear(feat_dim, hidden_size)
        self.visn_layer_norm = BertLayerNorm(hidden_size, eps=1e-9)

        self.dropout = nn.Dropout(drop)

    def forward(self, vis_input):
        feats = vis_input
        
        x = self.visn_fc(feats)
#         x = self.visn_layer_norm(x)
        output = x

        output = self.dropout(output)
        return output


class NoPosLXRTEncoder(nn.Module):
    def __init__(self, visual_feat_dim=2048, drop=0.0, l_layers=9, x_layers=5, r_layers=5, num_attention_heads=4, hidden_size=3072,intermediate_size=2048):
        super().__init__()

        # Obj-level image embedding layer
        self.visn_fc = NoPosVisualFeatEncoder(visual_feat_dim=visual_feat_dim, hidden_size=hidden_size, drop=0.0)

        # Number of layers
        self.num_l_layers = l_layers
        self.num_x_layers = x_layers
        self.num_r_layers = r_layers
        print("LXRT encoder with %d l_layers, %d x_layers, and %d r_layers." %
              (self.num_l_layers, self.num_x_layers, self.num_r_layers))

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.layer = nn.ModuleList(
            [BertLayer(num_attention_heads=num_attention_heads, hidden_size=hidden_size, drop=drop, intermediate_size=intermediate_size) for _ in range(self.num_l_layers)]
        )
        self.x_layers = nn.ModuleList(
            [LXRTXLayer(num_attention_heads=num_attention_heads, hidden_size=hidden_size, drop=drop, intermediate_size=intermediate_size) for _ in range(self.num_x_layers)]
        )
        self.r_layers = nn.ModuleList(
            [BertLayer(num_attention_heads=num_attention_heads, hidden_size=hidden_size, drop=drop, intermediate_size=intermediate_size) for _ in range(self.num_r_layers)]
        )
        
        
    def forward(self, lang_feats, lang_attention_mask,
                visn_feats, visn_attention_mask=None, logger=False):
        # Run visual embedding layer
        # Note: Word embedding layer was executed outside this module.
        #       Keep this design to allow loading BERT weights.
        
        if visn_feats != None:
            visn_feats = self.visn_fc(visn_feats)

            # Run language layers
            count = 0
            check = False
            for layer_module in self.layer:
                lang_feats = layer_module(lang_feats, lang_attention_mask)

            # Run relational layers
            for layer_module in self.r_layers:
                visn_feats = layer_module(visn_feats, visn_attention_mask)

            for layer_module in self.x_layers:
                lang_feats, visn_feats = layer_module(lang_feats, lang_attention_mask,
                                                      visn_feats, visn_attention_mask)
        else:
            for layer_module in self.layer:
                lang_feats = layer_module(lang_feats, lang_attention_mask)
            
        return lang_feats, visn_feats

