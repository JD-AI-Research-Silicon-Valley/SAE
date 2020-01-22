from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
#from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

import argparse
import csv
import logging
import os
import random
import sys
import pickle
import json
import time
import functools
import code

from collections import OrderedDict
import numpy as np
import torch

#
#CLS = 101
#SEP = 102

#Find and upweigh the documents with the supporting facts?
#Find the document with the answer span in it as well, and doubly upweigh it 2->1->0
#This should be used in a regression type setting?
def find_facts(ex):
    doc_dict=OrderedDict()
    for i, doc in enumerate(ex['context']):
        doc_dict[doc[0]] = [i,0]
    for i, id_sent in enumerate(ex['supporting_facts']):
        if doc_dict.get(id_sent[0]) is not None:
            doc_dict[id_sent[0]][1]=1
            if ex['answer'] not in ['yes','no']:
                index = doc_dict[id_sent[0]][0]
                if ex['answer'] in ''.join(ex['context'][index][1]):
                    doc_dict[id_sent[0]][1]=2
    return doc_dict

def find_test_facts(ex):
    doc_dict=OrderedDict()
    for i, doc in enumerate(ex['context']):
        if doc[0] == ex['hop1_query'] or doc[0] == ex['hop2_query']:
            doc_dict[doc[0]] = [i,1]
        else:
            doc_dict[doc[0]] = [i,0]
    return doc_dict



# sents
# docs
# offsets
# question
# labels
def get_features(data_path, tokenizer, use_mini=False, max_seq_len=512):
    max_sent_len=0
    with open(data_path, 'r') as f:
        d = json.load(f)
    all_labels, all_offsets, all_ques, all_docs, all_title = [],[],[],[],[]
    for i, ex in enumerate(d):
        if use_mini and i == 128:
            break
        sent_offset=0
        offsets=[] 
        example_docs=[]
        ques_tokens = tokenizer.tokenize(ex['question'])
        ques = tokenizer.convert_tokens_to_ids(ques_tokens)
        for j, doc in enumerate(ex['context']):
            title = doc[0]
            docs=[]
            for k, sent in enumerate(doc[1]):
                sent_toks = tokenizer.tokenize(sent)
                if len(sent_toks) > 512:
                    print(len(sent_toks))
                    print(sent)
                    print(sent_toks)
                if len(sent_toks) > max_sent_len:
                    max_sent_len = len(sent_toks)
                sent_ids = tokenizer.convert_tokens_to_ids(sent_toks)
                docs.append(sent_ids)
                sent_offset+=len(sent_toks)
                offsets.append(sent_offset)
            example_docs.append(docs)
        labels=[]
        #if 'answer' in ex:
        if False:
            facts = find_facts(ex)
            for key, item in facts.items():
                labels.append(item[-1])
        all_labels.append(labels)
        all_offsets.append(offsets)
        all_ques.append(ques)
        all_docs.append(example_docs)
        all_title.append(title)
        if i%100==0:
            print('Max Sent Length ', i, max_sent_len)
    return [all_ques, all_docs, all_offsets, all_labels, all_title]

def get_test_features(data_path, tokenizer, use_mini=False, max_seq_len=512):
    max_sent_len=0
    with open(data_path, 'r') as f:
        d = json.load(f)
    all_labels, all_offsets, all_ques, all_docs, all_title = [],[],[],[],[]
    for i, ex in enumerate(d):
        if use_mini and i == 64:
            break
        sent_offset=0
        offsets=[] 
        example_docs=[]
        ques_tokens = tokenizer.tokenize(ex['question'])
        ques = tokenizer.convert_tokens_to_ids(ques_tokens)
        for j, doc in enumerate(ex['context']):
            title = doc[0]
            docs=[]
            for k, sent in enumerate(doc[1]):
                sent_toks = tokenizer.tokenize(sent)
                if len(sent_toks) > 512:
                    print(len(sent_toks))
                    print(sent)
                    print(sent_toks)
                if len(sent_toks) > max_sent_len:
                    max_sent_len = len(sent_toks)
                sent_ids = tokenizer.convert_tokens_to_ids(sent_toks)
                docs.append(sent_ids)
                sent_offset+=len(sent_toks)
                offsets.append(sent_offset)
            example_docs.append(docs)
        labels=[]
        facts = find_test_facts(ex)
        for key, item in facts.items():
            labels.append(item[-1])
        all_labels.append(labels)
        all_offsets.append(offsets)
        all_ques.append(ques)
        all_docs.append(example_docs)
        all_title.append(title)
        if i%100==0:
            print('Max Sent Length ', i, max_sent_len)
    return [all_ques, all_docs, all_offsets, all_labels, all_title]

#Finish writing out train/dev split on this
def preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_mini', action='store_true', help='Load in a mini set for debug')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum BERT Sequence Length')
    parser.add_argument('--do_lower_case', action='store_true', help='Lower case BERT')
    parser.add_argument('--use_peng', action='store_true', help='Load in Peng IR results')
    parser.add_argument('--orig_train_file', type=str, default='hotpot_train_v1.1.json', help='Train File Name')
    parser.add_argument('--orig_dev_file', type=str, default='hotpot_dev_distractor_v1.json', help='Dev File Name')
    parser.add_argument('--peng_train_file', type=str, default='hotpot_dev_distractor_v1.json', help='Peng Train File Name')
    parser.add_argument('--peng_dev_file', type=str, default='hotpot_dev_distractor_v1.json', help='Peng Dev File Name')
    parser.add_argument('--data_dir', type=str, default='/mnt/cephfs2/asr/users/kevin.huang/hotpot/data',
                        help='Load in a mini set for debug')
    parser.add_argument('--name', type=str, default='',
                        help='name for the saved file')
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help='Bert Model to use for tokenization')
    args = parser.parse_args()
    
    args.do_lower_case = True if 'uncased' in args.bert_model else False
    
    print('ARGS LOWER CASE: ', args.do_lower_case)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    train_file = args.peng_train_file if args.use_peng else args.orig_train_file
    dev_file = args.peng_dev_file if args.use_peng else args.orig_dev_file
    train_features = get_features(os.path.join(args.data_dir, train_file), tokenizer, args.use_mini)
    dev_features = get_features(os.path.join(args.data_dir, dev_file), tokenizer, args.use_mini)

    args.name = 'peng' if args.use_peng else 'orig'
    train_name = 'train_'+args.name+'_'+args.bert_model+'.pkl'
    dev_name = 'dev_'+args.name+'_'+args.bert_model+'.pkl'
    if args.use_mini:
        dev_name = 'mini_'+dev_name
        train_name = 'mini_'+train_name
    with open(train_name, 'wb') as f:
        pickle.dump(train_features, f)
    with open(dev_name, 'wb') as f:
        pickle.dump(dev_features, f)
    if args.use_peng == True:
        test_name = 'test_peng_bert-base-uncased.pkl'
        peng_dev_features = get_test_features(os.path.join(args.data_dir, 'qa_input.json'), tokenizer, False)
        with open(test_name, 'wb') as f:
            pickle.dump(peng_dev_features, f)
if __name__ == "__main__":
    preprocess()
