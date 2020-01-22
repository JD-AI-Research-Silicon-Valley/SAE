
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
""" Load SQuAD dataset. """

from __future__ import absolute_import, division, print_function

import json, sys, string, re
import logging
import math
import collections
from io import open

from pytorch_transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

# Required by XLNet evaluation method to compute optimal threshold (see write_predictions_extended() method)

logger = logging.getLogger(__name__)

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def check_overlap(sent1, sent2):
    for ner1 in sent1:
        for ner2 in sent2:
            if ner1[0] == ner2[0]:
                return True
    
    return False

def check_ques(sent1, sent2, ques):
    for ner1 in sent1:
        for ner2 in sent2:
            #print(ner1[0], ner2[0], ques)
            #if ner1[0] != ner2[0] and ner1[0] in ques and ner2[0] in ques:
            if ner1[0] in ques and ner2[0] in ques: # no matter same or not
                return True
    return False

def wd_edge_list(data):
    """with-document edge list
    
    Arguments:
        data {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    wd_edge_list = []
    for s1i, s1 in enumerate(data): # even for doc title, odd for doc sents
        for s2i, s2 in enumerate(data):
            if s1i!=s2i and s1.doc_id == s2.doc_id:
                wd_edge_list.append([s1i, s2i])
    return wd_edge_list

def ad_edge_list(data):
    """with-document edge list
    
    Arguments:
        data {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    ad_edge_list = []
    for s1i, s1 in enumerate(data): # even for doc title, odd for doc sents
        for s2i, s2 in enumerate(data) :
            if s1.doc_id != s2.doc_id and check_overlap(s1.sent_ner, s2.sent_ner):
                ad_edge_list.append([s1i, s2i])
    return ad_edge_list

def ques_edge_list(data):
    """connections with question NER and NP as bridge
    
    Arguments:
        data {[type]} -- [description]
    """
    try:
        ner_in_q = [item[0] for item in data[0].question_ner]
    except:
        print(data)
        exit()
    ques_edge_list = []
    for di_idx, di in enumerate(data):
        for dj_idx, dj in enumerate(data):
            if check_ques(di.sent_ner, dj.sent_ner, ner_in_q) and (di.doc_id != dj.doc_id):
                ques_edge_list.append([di_idx, dj_idx])
    return ques_edge_list


class HotpotExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 question_ner,
                 sent_text,
                 sent_ner,
                 doc_id,
                 doc_title,
                 sent_id,
                 answer_text,
                 answer_type,
                 is_sp):
        self.qas_id = qas_id
        self.question_text = question_text
        self.question_ner = question_ner
        self.sent_text = sent_text
        self.sent_ner = sent_ner
        self.doc_id = doc_id
        self.doc_title = doc_title,
        self.sent_id = sent_id
        self.answer_text = answer_text
        self.answer_type = answer_type
        self.is_sp = is_sp

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += "\n sent_text: [%s]" % (" ".join(self.sent_text))
        s += "\n sent_ner: [%s]" % (" ".join(self.sent_ner))
        s += "\n is_sp: [%s]" % (self.is_sp)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 orig_tokens,
                 token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids,
                 p_mask,
                 cls_index,
                 wd_edges,
                 ques_edges,
                 ad_edges,
                 graph_mask,
                 sent_start,
                 sent_end,
                 start_position=None,
                 end_position=None,
                 sp_label = None,
                 answer_type = None
                ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.orig_tokens = orig_tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.p_mask = p_mask
        self.cls_index = cls_index
        self.wd_edges = wd_edges
        self.ques_edges = ques_edges
        self.ad_edges = ad_edges
        self.sent_start = sent_start
        self.sent_end = sent_end
        self.start_position = start_position
        self.end_position = end_position
        self.sp_label = sp_label
        self.answer_type = answer_type
        self.graph_mask = graph_mask


def read_train_examples(input_file, ner_file, is_gold, squad_num=0):
    """Read a SQuAD json file into a list of SquadExample."""

    # hotpotqa num
    hotpot_num = 90447
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    with open(ner_file, 'r', encoding='utf-8') as reader:
        ner_data = json.load(reader)

    if squad_num > 0:
        input_data = input_data[:hotpot_num + squad_num]
        ner_data = ner_data[:hotpot_num+squad_num]

    # print(len(input_data))
    # print(len(ner_data))
    # data_ids = [entry['_id'] for entry in input_data]
    # ner_ids = [entry['_id'] for entry in ner_data]
    # print(len(set(data_ids)))
    # print(len(set(ner_ids)))
    # exit()

    def title_index(titles,t):
        for idx, title in enumerate(titles):
            if t == title:
                return idx       
        return -1

    def answer_map(ans):
        if ans == "no":
            return 0
        elif ans == "yes":
            return 1
        else:
            return 2

    examples, max_sent_num = [], 0
    for idx, entry in enumerate(input_data):

        doc_title, doc_texts = [], []
        for paragraph in entry["context"]:
            doc_title.append(paragraph[0])
            doc_text = []
            for sent in paragraph[1]:
                doc_text.append(sent)
            doc_texts.append(doc_text)

        paragraphs = []
        doc_id = set()
        sp_title = [] # for debug
        for sp in entry["supporting_facts"]:
            sp_title.append(sp[0])
            doc_id.add( title_index(doc_title, sp[0]) )
        doc_id = list(doc_id)
        sp_doc_title = [doc_title[x] for x in doc_id]
        for id in doc_id:
            if id != -1:
                try:
                    paragraphs.append([doc_title[id], doc_texts[id] ] )
                except:
                    continue

        max_sent_num = max(max_sent_num, sum([len(x[1]) for x in paragraphs]))

        # get updated supporting facts indices
        sp_idx = []
        for sp in entry["supporting_facts"]:
            if sp[1] < len(paragraphs[sp_doc_title.index(sp[0])][1]):
                sp_idx.append([sp_doc_title.index(sp[0]), sp[1]])

        # get updated ner data
        tmp_ner_data = []
        tmp_np_data = []
        for ni in doc_id:
            tmp_ner_data.append(ner_data[idx]['context_ner_pos'][1::2][ni])
            tmp_np_data.append(ner_data[idx]['context_noun_pos'][1::2][ni])
        
        # check consistency        
        for ni, ner in enumerate(tmp_ner_data):
            try:
                assert(len(ner) == len(paragraphs[ni][1]))
            except:
                print(ner)
                print(paragraphs[ni][1])
                sys.exit()
        
        ner_data[idx]['context_ner_pos'] = tmp_ner_data
        ner_data[idx]['context_noun_pos'] = tmp_np_data

        qas_id = entry['_id']
        question_text = entry['question']
        question_ner = [item for item in ner_data[idx]['question_ner_pos']] + [item for item in ner_data[idx]['question_noun_pos']]
        
        answer_text = entry['answer']
        answer_type = answer_map(answer_text)

        example_docs, doc_title = [], []
        for di, doc in enumerate(paragraphs):
            doc_title = doc[0]
            for si, sent in enumerate(doc[1]):
                is_sp = [di, si] in sp_idx
                example_docs.append(
                    HotpotExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    question_ner = question_ner,
                    sent_text=sent,
                    sent_ner = ner_data[idx]['context_ner_pos'][di][si] + ner_data[idx]['context_noun_pos'][di][si],
                    doc_id = di,
                    doc_title = doc_title,
                    sent_id = si,
                    answer_text=answer_text,
                    answer_type = answer_type,
                    is_sp = is_sp)
                )
        examples.append(example_docs)
    return examples, max_sent_num

def read_eval_examples(input_file, ner_file):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    with open(ner_file, 'r', encoding='utf-8') as reader:
        ner_data = json.load(reader)

    examples, max_sent_num = [], 0
    for idx, entry in enumerate(input_data):
        assert(entry['_id'] == ner_data[idx]['_id'])

        # get updated ner data
        context_ner = []
        for doc in entry['context']:
            flag = 0
            tmp_ner = []
            for ni, doc_title_ner in enumerate(ner_data[idx]['context_ner_pos'][0::2]):
                if len(doc_title_ner) != 0:
                    if doc[0] == doc_title_ner[0][0] and len(doc[1]) == len(ner_data[idx]['context_ner_pos'][1::2][ni]):
                        cur_ner = ner_data[idx]['context_ner_pos'][1::2][ni]
                        cur_np = ner_data[idx]['context_noun_pos'][1::2][ni]
                        for si in range(len(cur_ner)):
                            tmp_ner.append(cur_ner[si] + cur_np[si])
                        flag = 1 # find the ners of the doc
                        break
            if flag == 0: # can't find ners of the current doc
                tmp_ner = [[] for i in range(len(doc[1]))]
            context_ner.append(tmp_ner)
            try:
                assert(len(tmp_ner) == len(doc[1]))
            except:
                print(len(tmp_ner), tmp_ner)
                print(len(doc[1]), doc)
                exit()
        
        question_ner = [item for item in ner_data[idx]['question_ner_pos']] + [item for item in ner_data[idx]['question_noun_pos']]

        example_docs = []
        if len(entry['context']) > 0:
            for di, doc in enumerate(entry['context']):
                for si, sent in enumerate(doc[1]):
                    example_docs.append(
                        HotpotExample(
                        qas_id=entry['_id'],
                        question_text=entry['question'],
                        question_ner = question_ner,
                        sent_text=sent,
                        sent_ner = context_ner[di][si],
                        doc_id = di,
                        doc_title = doc[0],
                        sent_id = si,
                        answer_text=None,
                        answer_type = None,
                        is_sp = None))
            max_sent_num = max(max_sent_num, len(example_docs))
        else:
            null_example = HotpotExample(
                        qas_id=entry['_id'],
                        question_text=entry['question'],
                        question_ner = question_ner,
                        sent_text=" ",
                        sent_ner = [["", [], 0, 0]],
                        doc_id = 0,
                        doc_title = " ",
                        sent_id = 0,
                        answer_text=None,
                        answer_type = None,
                        is_sp = None)
            example_docs.append(null_example)
        examples.append(example_docs)
    return examples, max_sent_num

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    tokens_a_del = 0
    tokens_b_del = 0
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            tokens_a_del += 1
        else:
            tokens_b.pop()
            tokens_b_del += 1
    
    return tokens_a_del, tokens_b_del

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', 
                                 sep_token_extra=False, pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """Loads a data file into a list of `InputBatch`s."""

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):

        all_tokens = []
        all_input_ids = []
        all_segment_ids = []
        all_input_mask = []
        all_pmask = []
        all_graph_mask = []
        sent_start = []
        sent_end = []

        if len(example) > 0:

            answer_type = None
            sp_label = None
            start_position = None
            end_position = None
            if is_training:
                answer_type = example[0].answer_type
                sp_label = []

            query_tokens = tokenizer.tokenize(example[0].question_text)
            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]

            sent_tokens = []
            sent_text = " ".join([sent.sent_text.strip() for sent in example]).strip()
            cur_pos = 0
            for si, sent in enumerate(example):
                sent_start.append(cur_pos)
                cur_tokens = tokenizer.tokenize(sent.sent_text)
                sent_tokens.extend(cur_tokens)
                sent_end.append(cur_pos + len(cur_tokens))
                cur_pos += len(cur_tokens)
                all_graph_mask.append(1)
                if is_training:
                    sp_label.append(int(sent.is_sp))

            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in sent_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            if is_training:
                char_start_position, char_end_position = None, None 
                for i in range(len(sent_text)):
                    if normalize_answer(sent_text[i:i+len(example[0].answer_text)]) == normalize_answer(example[0].answer_text):
                        char_start_position = char_to_word_offset[i]
                        char_end_position = char_to_word_offset[i + len(example[0].answer_text) - 1]
                        break

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            q_del, s_del = _truncate_seq_pair(query_tokens, all_doc_tokens, max_seq_length - 3 - int(sep_token_extra))
            for si in range(len(sent_start)):
                sent_start[si] += len(query_tokens) + 2 + int(sep_token_extra)
                sent_end[si] += len(query_tokens) + 2 + int(sep_token_extra)

            tok_start_position = None
            tok_end_position = None
            if is_training:
                if char_start_position is not None:
                    tok_start_position = orig_to_tok_index[char_start_position]
                    if char_end_position < len(doc_tokens) - 1:
                        tok_end_position = orig_to_tok_index[char_end_position + 1] - 1
                    else:
                        tok_end_position = len(all_doc_tokens) - 1
                    (tok_start_position, tok_end_position) = _improve_answer_span(
                        all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                        example[0].answer_text)
                else:
                    tok_start_position = 0 if not cls_token_at_end else -1
                    tok_end_position = 0 if not cls_token_at_end else -1 

            tokens = []
            segment_ids = []
            p_mask = []

            if not cls_token_at_end:
                tokens.append(tokenizer.cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0

            # Query
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            # SEP token
            tokens.append(tokenizer.sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [tokenizer.sep_token]
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            token_to_orig_map = {}
            for i in range(len(all_doc_tokens)):
                token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
                tokens.append(all_doc_tokens[i])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)

            # SEP token
            tokens.append(tokenizer.sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(tokenizer.cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            all_tokens = tokens
            all_input_ids = input_ids
            all_input_mask = input_mask
            all_segment_ids = segment_ids
            all_pmask = p_mask
            
            if is_training:
                if answer_type == 2:
                    start_position = tok_start_position + len(query_tokens) + 2 + int(sep_token_extra)
                    # if start_position >= max_seq_length:
                    #     logger.info("example_index: %s" % (example_index))
                    #     logger.info("tokens: %s" % " ".join(tokens))
                    #     logger.info("answer: %s" % (example[0].answer_text))
                    #     logger.info("start_position: %d" % (start_position))
                    #     logger.info("end_position: %d" % (end_position)) 
                    #     exit() 
                    start_position = start_position if start_position < max_seq_length else cls_index
                    end_position = tok_end_position + len(query_tokens) + 2 + int(sep_token_extra)
                    end_position = end_position if end_position < max_seq_length else cls_index
                else:
                    start_position = cls_index
                    end_position = cls_index

            if len(example) > 0:
                wd_edges, ques_edges, ad_edges = wd_edge_list(example), ques_edge_list(example), ad_edge_list(example)
            else:
                wd_edges, ques_edges, ad_edges = [], [], []

            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info(
                    "p_mask: %s" % " ".join([str(x) for x in p_mask]))
                if is_training:
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info( "answer: %s" % (" ".join(tokens[start_position:(end_position + 1)])))
                    logger.info("answer type: %s" % (answer_type))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    tokens=all_tokens,
                    orig_tokens = doc_tokens,
                    token_to_orig_map = token_to_orig_map,
                    input_ids=all_input_ids,
                    input_mask=all_input_mask,
                    segment_ids=all_segment_ids,
                    p_mask = all_pmask,
                    cls_index = cls_index,
                    start_position=start_position,
                    end_position=end_position,
                    sent_start = sent_start,
                    sent_end = sent_end,
                    sp_label = sp_label,
                    answer_type = answer_type,
                    wd_edges = wd_edges,
                    ques_edges = ques_edges,
                    ad_edges = ad_edges,
                    graph_mask = all_graph_mask))
        else:
            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    tokens=tokenizer.cls_token + ['q_placeholder']*max_query_length + [tokenizer.sep_token]*(1+int(sep_token_extra)) + ['c_placeholder']*(max_seq_length-max_query_length-3-int(sep_token_extra)) + tokenizer.sep_token,
                    orig_tokens = ['placeholder']*max_seq_length,
                    token_to_orig_map = {i:i for i in range(max_seq_length)},
                    input_ids=[0]*max_seq_length,
                    input_mask=[1]*max_seq_length,
                    segment_ids=[0]*(max_query_length+2) + [1]*(max_seq_length-max_query_length-2),
                    p_mask = [1]*max_seq_length,
                    cls_index = cls_index,
                    start_position=None if is_training else 0,
                    end_position=None if is_training else 0,
                    sent_start = [0]*10,
                    sent_end = [1]*10,
                    sp_label = None if is_training else [],
                    answer_type = None if is_training else 2,
                    wd_edges = [],
                    ques_edges = [],
                    ad_edges = [],
                    graph_mask = [1]*10)) 

        unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = feature.orig_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))
                
            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest)==1:
                nbest.insert(0,
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example[0].qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example[0].qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example[0].qas_id] = ""
            else:
                all_predictions[example[0].qas_id] = best_non_null_entry.text
        all_nbest_json[example[0].qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


# For XLNet (and XLM which uses the same head)
RawResultExtended = collections.namedtuple("RawResultExtended",
    ["unique_id", "start_top_log_probs", "start_top_index",
     "end_top_log_probs", "end_top_index", "cls_logits"])


def write_predictions_extended(all_examples, all_features, all_results, n_best_size,
                                max_answer_length, output_prediction_file,
                                output_nbest_file,
                                output_null_log_odds_file, orig_data_file,
                                start_n_top, end_n_top, version_2_with_negative,
                                tokenizer, verbose_logging):
    """ XLNet write prediction logic (more complex than Bert's).
        Write final predictions to the json file and log-odds of null if needed.

        Requires utils_squad_evaluate.py
    """
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index",
        "start_log_prob", "end_log_prob"])

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])

    logger.info("Writing predictions to: %s", output_prediction_file)
    # logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_top_log_probs[i]
                    start_index = result.start_top_index[i]

                    j_index = i * end_n_top + j

                    end_log_prob = result.end_top_log_probs[j_index]
                    end_index = result.end_top_index[j_index]

                    print(start_index, end_index)

                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
                        continue

                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_log_prob=start_log_prob,
                            end_log_prob=end_log_prob))

            exit()

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob),
            reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            # XLNet un-tokenizer
            # Let's keep it simple for now and see if we need all this later.
            # 
            # tok_start_to_orig_index = feature.tok_start_to_orig_index
            # tok_end_to_orig_index = feature.tok_end_to_orig_index
            # start_orig_pos = tok_start_to_orig_index[pred.start_index]
            # end_orig_pos = tok_end_to_orig_index[pred.end_index]
            # paragraph_text = example.paragraph_text
            # final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

            # Previously used Bert untokenizer
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, tokenizer.do_lower_case,
                                        verbose_logging)

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="", start_log_prob=-1e6,
                end_log_prob=-1e6))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None

        score_diff = score_null
        scores_diff_json[example.qas_id] = score_diff
        # note(zhiliny): always predict best_non_null_entry
        # and the evaluation script will search for the best threshold
        all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    with open(orig_data_file, "r", encoding='utf-8') as reader:
        orig_data = json.load(reader)["data"]

    qid_to_has_ans = make_qid_to_has_ans(orig_data)
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw = get_raw_scores(orig_data, all_predictions)
    out_eval = {}

    find_all_best_thresh_v2(out_eval, all_predictions, exact_raw, f1_raw, scores_diff_json, qid_to_has_ans)

    return out_eval


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
