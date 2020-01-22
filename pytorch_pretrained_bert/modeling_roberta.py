# coding=utf-8
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
"""PyTorch RoBERTa model. """

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from .modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu, BertSelfAttention
from .modeling_utils import PreTrainedModel, prune_linear_layer, SequenceSummary, PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits
from .configuration_roberta import RobertaConfig
from .file_utils import add_start_docstrings

#from sentGNN_DGL_BiDAF import seqGNN_DGL_BiDAF

logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}

class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__(config)
        self.padding_idx = 1
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size,
                                                padding_idx=self.padding_idx)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            # Position numbers begin at padding_idx+1. Padding symbols are ignored.
            # cf. fairseq's `utils.make_positions`
            position_ids = torch.arange(self.padding_idx+1, seq_length+self.padding_idx+1, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return super(RobertaEmbeddings, self).forward(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)

# class RobertaEmbeddings(nn.Module):
#     """
#     Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
#     """
#     def __init__(self, config):
#         super(RobertaEmbeddings, self).__init__()
#         self.padding_idx = config.padding_idx
#         self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, 
#                                             padding_idx=config.padding_idx)
#         self.position_embeddings = nn.Embedding(self.padding_idx + config.max_position_embeddings + 1,
#                                                 config.hidden_size,
#                                                 padding_idx=config.padding_idx)
#         print(self.padding_idx + config.max_position_embeddings + 1)
#         exit()
#         if config.type_vocab_size > 0:
#             self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
#         else:
#             self.token_type_embeddings = None
#         self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def forward(self, input_ids, token_type_ids=None, position_ids=None):
#         if position_ids is None:
#             # Position numbers begin at padding_idx+1. Padding symbols are ignored.
#             # cf. fairseq's `utils.make_positions`
#             # position_ids.masked_fill_(mask,self.padding_idx)
#             # fairseq version for ONNX and XLA compatiblity
#             mask = input_ids.ne(self.padding_idx).int()
#             position_ids = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx

#         if self.token_type_embeddings is not None and self.token_type_ids is None:
#             token_type_ids = torch.zeros_like(input_ids,dtype=torch.long, device=input_ids.device)

#         words_embeddings = self.word_embeddings(input_ids)
#         position_embeddings = self.position_embeddings(position_ids)
#         embeddings = words_embeddings + position_embeddings
#         if self.token_type_embeddings is not None:
#             token_type_embeddings = self.token_type_embeddings(token_type_ids)
#             embeddings = embeddings + token_type_embeddings
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         exit()
#         return embeddings


ROBERTA_START_DOCSTRING = r"""    The RoBERTa model was proposed in
    `RoBERTa: A Robustly Optimized BERT Pretraining Approach`_
    by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer,
    Veselin Stoyanov. It is based on Google's BERT model released in 2018.
    
    It builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining
    objective and training with much larger mini-batches and learning rates.
    
    This implementation is the same as BertModel with a tiny embeddings tweak as well as a setup for Roberta pretrained 
    models.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`RoBERTa: A Robustly Optimized BERT Pretraining Approach`:
        https://arxiv.org/abs/1907.11692

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~pytorch_transformers.RobertaConfig`): Model configuration class with all the parameters of the 
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~pytorch_transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, RoBERTa input sequence should be formatted with <s> and </s> tokens as follows:

            (a) For sequence pairs:

                ``tokens:         <s> Is this Jacksonville ? </s> </s> No it is not . </s>``

            (b) For single sequences:

                ``tokens:         <s> the dog is hairy . </s>``

            Fully encoded sequences or sequence pairs can be obtained using the RobertaTokenizer.encode function with 
            the ``add_special_tokens`` parameter set to ``True``.

            RoBERTa is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            See :func:`pytorch_transformers.PreTrainedTokenizer.encode` and
            :func:`pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""

@add_start_docstrings("The bare RoBERTa Model transformer outputing raw hidden-states without any specific head on top.",
                      ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class RobertaModel(BertModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaModel, self).__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        if input_ids[:, 0].sum().item() != 0:
            logger.warning("A sequence with no special tokens has been passed to the RoBERTa model. "
                           "This model requires special tokens in order to work. "
                           "Please specify add_special_tokens=True in your encoding.")
        return super(RobertaModel, self).forward(input_ids, token_type_ids, attention_mask, position_ids, head_mask)


@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top. """,
    ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class RobertaForMaskedLM(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMaskedLM, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head.decoder, self.roberta.embeddings.word_embeddings)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, position_ids=None,
                head_mask=None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super(RobertaLMHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias

        return x


@add_start_docstrings("""RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer 
    on top of the pooled output) e.g. for GLUE tasks. """,
    ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class RobertaForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RoertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)



class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

@add_start_docstrings("""RoBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class RobertaForQuestionAnswering(BertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **is_impossible**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels whether a question has an answer or no answer (SQuAD 2.0)
        **cls_index**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the classification token to use as input for computing plausibility of the answer.
        **p_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Optional mask of tokens which can't be in answers (e.g. [CLS], [PAD], ...).
            1.0 means token should be masked. 0.0 mean token is not masked.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned if both ``start_positions`` and ``end_positions`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification losses.
        **start_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top)``
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        **start_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top)``
            Indices for the top config.start_n_top start token possibilities (beam-search).
        **end_top_log_probs**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
            Log probabilities for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        **end_top_index**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.LongTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``
            Indices for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        **cls_logits**: (`optional`, returned if ``start_positions`` or ``end_positions`` is not provided)
            ``torch.FloatTensor`` of shape ``(batch_size,)``
            Log probabilities for the ``is_impossible`` label of the answers.
        **mems**:
            list of ``torch.FloatTensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            if config.mem_len > 0 else tuple of None. Can be used to speed up sequential decoding and attend to longer context.
            See details in the docstring of the `mems` input above.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
        model = XLMForQuestionAnswering.from_pretrained('xlnet-large-cased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])
        outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        loss, start_scores, end_scores = outputs[:2]

    """

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.answer_class = PoolerAnswerClass(config)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None,
                start_positions=None, end_positions=None, cls_index=None, is_impossible=None, p_mask=None,
                head_mask=None):
        
        # Roberta doesn't use token_type_ids cause there is no NSP task
        token_type_ids = torch.zeros_like(token_type_ids).to(input_ids.device)

        transformer_outputs = self.roberta(input_ids, token_type_ids=token_type_ids,
                                               attention_mask=attention_mask, position_ids=position_ids,
                                               head_mask=head_mask)
        hidden_states = transformer_outputs[0]
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + transformer_outputs[2:] 

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if cls_index is not None and is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)

                # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
                total_loss += cls_loss * 0.5

            outputs = (total_loss,) + outputs 

        # return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits
        # or (if labels are provided) (total_loss,)
        return outputs

class gcnLayer(nn.Module):
    def __init__(self, input_dim, proj_dim=512, dropout=0.1, num_hop=3, gcn_num_rel=3, batch_norm=False, edp=0.0):
        super(gcnLayer, self).__init__()
        self.proj_dim = proj_dim
        self.num_hop = num_hop
        self.gcn_num_rel = gcn_num_rel
        self.dropout = dropout
        self.edge_dropout = nn.Dropout(edp)

        for i in range(gcn_num_rel):
            setattr(self, "fr{}".format(i+1), nn.Sequential(nn.Linear(input_dim, proj_dim), nn.Dropout(dropout, inplace=False)))

        self.fs = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.Dropout(dropout, inplace=False))

        self.fa = nn.Sequential(nn.Linear(input_dim + proj_dim, proj_dim))

        self.act = GeLU()

    def forward(self, input, input_mask, adj):
        # input: bs x max_nodes x node_dim
        # input_mask: bs x max_nodes
        # adj: bs x 3 x max_nodes x max_nodes
        # num_layer: number of layers; note that the parameters of all layers are shared

        cur_input = input.clone()

        for i in range(self.num_hop):
            # integrate neighbor information
            nb_output = torch.stack([getattr(self, "fr{}".format(i+1))(cur_input) for i in range(self.gcn_num_rel)],
                                    1) * input_mask.unsqueeze(-1).unsqueeze(1)  # bs x 2 x max_nodes x node_dim
            
            # apply different types of connections, which are encoded in adj matrix
            update = torch.sum(torch.matmul(self.edge_dropout(adj.float()),nb_output), dim=1, keepdim=False) + \
                     self.fs(cur_input) * input_mask.unsqueeze(-1)  # bs x max_node x node_dim

            # get gate values
            gate = torch.sigmoid(self.fa(torch.cat((update, cur_input), -1))) * input_mask.unsqueeze(
                -1)  # bs x max_node x node_dim

            # apply gate values
            cur_input = gate * self.act(update) + (1 - gate) * cur_input  # bs x max_node x node_dim

        return cur_input


class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class RobertaForHotpotQA(BertPreTrainedModel):

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, num_answer_type=3, num_hop = 3, num_rel = 2, no_gnn=False, gsn=False, edp=0.0, span_from_sp = False, sp_from_span = False):
        super(RobertaForHotpotQA, self).__init__(config)
        self.roberta = RobertaModel(config)
        #self.model_freeze()

        self.dropout = config.hidden_dropout_prob
        self.no_gnn = no_gnn
        self.gsn = gsn
        self.num_rel = num_rel
        self.config = config
        self.span_from_sp = span_from_sp
        self.sp_from_span = sp_from_span

        if not self.no_gnn:
            self.sp_graph = gcnLayer(config.hidden_size, config.hidden_size, num_hop=num_hop, gcn_num_rel=num_rel,edp=edp)

        self.hidden_size = int(config.hidden_size/2)

        self.sent_selfatt = nn.Sequential(nn.Linear(config.hidden_size, self.hidden_size), GeLU(), 
                        nn.Dropout(self.dropout), nn.Linear(self.hidden_size, 1))

        self.qa_outputs = nn.Sequential(nn.Linear(config.hidden_size, self.hidden_size), GeLU(), nn.Dropout(self.dropout),
                                       nn.Linear(self.hidden_size, 2))
        
        if self.span_from_sp:
            self.qa_outputs_from_sp = nn.Sequential(nn.Linear(config.hidden_size, self.hidden_size), GeLU(), nn.Dropout(self.dropout),
                                       nn.Linear(self.hidden_size, 2)) 

        self.sp_classifier = nn.Sequential(nn.Linear(config.hidden_size, self.hidden_size), GeLU(), nn.Dropout(self.dropout),
                                       nn.Linear(self.hidden_size, 1)) # input: graph embeddings

        self.num_answer_type = num_answer_type
        self.sfm = nn.Softmax(-1)
        self.answer_type_classifier = nn.Sequential(nn.Linear(config.hidden_size, self.hidden_size), GeLU(), nn.Dropout(self.dropout),
                                       nn.Linear(self.hidden_size, self.num_answer_type)) # input: pooling over graph embeddings of multiple supporting sentences
        #self.answer_type_classifier.half()

        self.init_weights()

    def attention(self, x, z):
        # x: batch_size X max_nodes X feat_dim
        # z: attention logits

        att = self.sfm(z).unsqueeze(-1) # batch_size X max_nodes X 1

        output = torch.bmm(att.transpose(1,2), x)

        return output

    def gen_mask(self, max_len, lengths, device):
        lengths = lengths.type(torch.LongTensor)
        num = lengths.size(0)
        vals = torch.LongTensor(range(max_len)).unsqueeze(0).expand(num, -1)+1 # +1 for masking out sequences with length 0
        mask = torch.gt(vals, lengths.unsqueeze(1).expand(-1, max_len)).to(device)
        return mask
    
    # self attentive pooling
    def do_selfatt(self, input, input_len, selfatt, span_logits = None):

        # input: max_len X batch_size X dim

        input_mask = self.gen_mask(input.size(0), input_len, input.device)

        att = selfatt.forward(input).squeeze(-1).transpose(0,1)
        att = att.masked_fill(input_mask, -9e15)
        if span_logits is not None:
            att = att + span_logits
        att_sfm = self.sfm(att).unsqueeze(1)

        # print(att_sfm[56:63,:,:])
        # exit()

        output = torch.bmm(att_sfm, input.transpose(0,1)).squeeze(1) # batchsize x dim

        return output

    def forward(self, input_ids, input_mask, segment_ids, adj_matrix, graph_mask, sent_start, 
                sent_end, position_ids = None, head_mask=None, p_mask=None,
                start_positions=None, end_positions=None, sp_label=None, all_answer_type=None, sent_sum_way='avg', span_loss_weight = 1.0):

        """
        input_ids: bs X num_doc X num_sent X sent_len
        token_type_ids: same size as input_ids
        attention_mask: same size as input_ids
        input_adj_matrix: bs X 3 X max_nodes X max_nodes
        input_graph_mask: bs X max_nodes
        """

        # Roberta doesn't use token_type_ids cause there is no NSP task
        segment_ids = torch.zeros_like(segment_ids).to(input_ids.device)

        # reshaping
        bs, sent_len = input_ids.size()
        max_nodes = adj_matrix.size(-1)


        sequence_output, cls_output = self.roberta(input_ids, token_type_ids=segment_ids,
                                               attention_mask=input_mask, position_ids=position_ids,
                                               head_mask=head_mask)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        feat_dim = cls_output.size(-1)

        # sentence extraction
        per_sent_len = sent_end - sent_start
        max_sent_len = torch.max(sent_end - sent_start)
        # print("Maximum sent length is {}".format(max_sent_len))
        sent_output = torch.zeros(bs, max_nodes, max_sent_len, feat_dim).to(input_ids.device)
        span_logits = start_logits + end_logits
        sent_span_logits = -9e15*torch.ones(bs,max_nodes,max_sent_len).to(input_ids.device)
        for i in range(bs):
            for j in range(max_nodes):
                if sent_end[i,j] <= sent_len:
                    if sent_start[i,j] != -1 and sent_end[i,j] != -1:
                        sent_output[i,j,:(sent_end[i,j]-sent_start[i,j]),:] = sequence_output[i,sent_start[i,j]:sent_end[i,j],:]
                        sent_span_logits[i,j,:(sent_end[i,j]-sent_start[i,j])] = span_logits[i,sent_start[i,j]:sent_end[i,j]]
                else:
                    if sent_start[i,j] < sent_len:
                        sent_output[i,j,:(sent_len-sent_start[i,j]),:] = sequence_output[i,sent_start[i,j]:sent_len,:] 
                        sent_span_logits[i,j,:(sent_len-sent_start[i,j])] = span_logits[i,sent_start[i,j]:sent_len]

        if self.gsn:
            sent_output_gsn = self.sp_graph(sent_output, per_sent_len, graph_mask, adj_matrix)

            # sent summarization
            if sent_sum_way == 'avg':
                sent_sum_output = sent_output_gsn.mean(dim=2)
            elif sent_sum_way == 'attn':
                if self.sp_from_span:
                    sent_sum_output = self.do_selfatt(sent_output_gsn.contiguous().view(bs*max_nodes,max_sent_len,self.config.hidden_size).transpose(0,1), \
                            per_sent_len.view(bs*max_nodes), self.sent_selfatt, sent_span_logits.view(bs*max_nodes,max_sent_len)).view(bs,max_nodes,-1)
                else:
                    sent_sum_output = self.do_selfatt(sent_output_gsn.contiguous().view(bs*max_nodes,max_sent_len,self.config.hidden_size).transpose(0,1), \
                            per_sent_len.view(bs*max_nodes), self.sent_selfatt).view(bs,max_nodes,-1)
            
            gcn_output = sent_sum_output

        else:
            # sent summarization
            if sent_sum_way == 'avg':
                sent_sum_output = sent_output.mean(dim=2)
            elif sent_sum_way == 'attn':
                if self.sp_from_span:
                    sent_sum_output = self.do_selfatt(sent_output.contiguous().view(bs*max_nodes,max_sent_len,self.config.hidden_size).transpose(0,1), \
                            per_sent_len.view(bs*max_nodes), self.sent_selfatt, sent_span_logits.view(bs*max_nodes,max_sent_len)).view(bs,max_nodes,-1)
                else:
                    sent_sum_output = self.do_selfatt(sent_output.contiguous().view(bs*max_nodes,max_sent_len,self.config.hidden_size).transpose(0,1), \
                            per_sent_len.view(bs*max_nodes), self.sent_selfatt).view(bs,max_nodes,-1)

            # graph reasoning
            if not self.no_gnn and self.num_rel > 0:
                gcn_output = self.sp_graph(sent_sum_output, graph_mask, adj_matrix) # bs X max_nodes X feat_dim
            else:
                #gcn_output = self.sp_graph(sent_sum_output, graph_mask, torch.zeros(bs,1,max_nodes,max_nodes).to(input_ids.device))
                gcn_output = sent_sum_output

        # sp sent classification
        sp_logits = self.sp_classifier(gcn_output).view(bs, max_nodes)
        sp_logits = torch.where(graph_mask > 0, sp_logits, -9e15*torch.ones_like(sp_logits).to(input_ids.device))
        
        # select top 10 sentences with highest logits and then recalculate start and end logits
        if self.span_from_sp:
            sel_sent = torch.where(torch.sigmoid(sp_logits) > 0.5, torch.ones_like(sp_logits).to(input_ids.device), 
                                    torch.zeros_like(sp_logits).to(input_ids.device))
            seq_output_from_sp = torch.zeros(bs, sent_len, feat_dim).to(input_ids.device)
            for i in range(bs):
                for j in range(max_nodes):
                    if sel_sent[i,j] == 1.0:
                        if sent_end[i,j] <= sent_len:
                            if sent_start[i,j] != -1 and sent_end[i,j] != -1:
                                seq_output_from_sp[i,sent_start[i,j]:sent_end[i,j],:] = sequence_output[i,sent_start[i,j]:sent_end[i,j],:]
                        else:
                            if sent_start[i,j] < sent_len:
                                seq_output_from_sp[i,sent_start[i,j]:sent_len,:] = sequence_output[i,sent_start[i,j]:sent_len,:]
            
            logits2 = self.qa_outputs_from_sp(seq_output_from_sp)
            start_logits2, end_logits2 = logits2.split(1, dim=-1)
            start_logits2 = start_logits2.squeeze(-1)
            end_logits2 = end_logits2.squeeze(-1)

            final_start_logits = start_logits + start_logits2
            final_end_logits = end_logits + end_logits2
        else:
            final_start_logits = start_logits
            final_end_logits = end_logits

        # answer type logits
        ans_type_logits = self.answer_type_classifier(self.attention(gcn_output, sp_logits)).squeeze(1)

        if start_positions is not None:
            return self.loss_func(final_start_logits, final_end_logits, start_positions, end_positions, sp_logits, sp_label, ans_type_logits, all_answer_type, span_loss_weight), \
                            start_logits, end_logits, sp_logits, ans_type_logits
        else:
            return (start_logits, end_logits, sp_logits, ans_type_logits)

    def span_loss(self, start_logits, end_logits, start_positions, end_positions):
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss
    
    def loss_func(self, start_logits, end_logits, start_positions, end_positions, sp_logits, sp_label, ans_type_logits, ans_type_label, span_loss_weight):

        bce_crit = torch.nn.BCELoss()
        ce_crit = torch.nn.CrossEntropyLoss()

        # sp loss, binary cross entropy
        sp_loss = bce_crit(torch.sigmoid(sp_logits), sp_label.float())

        # answer type loss, cross entropy loss
        ans_loss = ce_crit(ans_type_logits, ans_type_label.long())

        # span loss
        span_loss = self.span_loss(start_logits, end_logits, start_positions, end_positions)

        return span_loss_weight*span_loss + sp_loss + ans_loss 
