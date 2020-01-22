import numpy as np
import torch
import scipy, time
from scipy.sparse import coo_matrix
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import json

def gen_adj_matrix(train_features, max_sent_num, adj_norm=False, wdedge=True, adedge=True, quesedge=True):
    
    def adj_proc(adj):
        adj = adj.numpy()
        d_ = adj.sum(-1)
        d_[np.nonzero(d_)] **=  -1
        return torch.tensor(adj * np.expand_dims(d_, -1))

    # edge list
    adj_matrix = []
    for f in train_features:
        adj_tmp = []
        if wdedge:
            if len(f.wd_edges) == 0:
                wd_edges_tmp = torch.zeros(max_sent_num, max_sent_num, dtype=torch.float)
            else:
                wd_edges_tmp = torch.tensor(coo_matrix((np.ones(len(f.wd_edges)), np.array(f.wd_edges).T),
                    shape=(max_sent_num, max_sent_num)).toarray(), dtype=torch.float)
            if adj_norm:
                adj_tmp.append(adj_proc(wd_edges_tmp))
            else:
                adj_tmp.append(wd_edges_tmp)
            # wd_edges_tmp = torch.zeros(max_sent_num, max_sent_num, dtype=torch.float)
            # if adj_norm:
            #     adj_tmp.append(adj_proc(wd_edges_tmp))
            # else:
            #     adj_tmp.append(wd_edges_tmp)
        if adedge:
            if len(f.ad_edges) == 0:
                ad_edges_tmp = torch.zeros(max_sent_num, max_sent_num, dtype=torch.float)
            else:
                ad_edges_tmp = torch.tensor(coo_matrix((np.ones(len(f.ad_edges)), np.array(f.ad_edges).T),
                    shape=(max_sent_num, max_sent_num)).toarray(), dtype=torch.float)
            if adj_norm:
                adj_tmp.append(adj_proc(ad_edges_tmp))
            else:
                adj_tmp.append(ad_edges_tmp)
        if quesedge:
            if len(f.ques_edges) == 0:
                ques_edges_tmp = torch.zeros(max_sent_num, max_sent_num, dtype=torch.float)
            else:
                ques_edges_tmp = torch.tensor(coo_matrix((np.ones(len(f.ques_edges)), np.array(f.ques_edges).T),
                    shape=(max_sent_num, max_sent_num)).toarray(), dtype=torch.float)
            if adj_norm:
                adj_tmp.append(adj_proc(ques_edges_tmp))
            else:
                adj_tmp.append(ques_edges_tmp)
        
        adj_matrix.append(torch.stack(adj_tmp,dim=0))
    adj_matrix = torch.stack(adj_matrix)

    return adj_matrix

class hotpotqa_joint_dataset(Dataset):
    
    '''
    Dataset class to read in scp and label files
    '''
    def __init__(self, features):

        self.data = features

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class MyCollator(object):
    
    def __init__(self, wd_edge = True, ques_edge = True, ad_edge=True, fcgnn = False, is_training=True):
        self.wd_edge = wd_edge
        self.ques_edge = ques_edge
        self.ad_edge = ad_edge
        self.fcgnn = fcgnn
        self.is_training = is_training

    def __call__(self, data_mb):

        start = time.time()

        def get_batch_stat(data_mb):
            max_sent_num = 0
            for d in data_mb:
                if len(d.sent_start) > max_sent_num:
                    max_sent_num = len(d.sent_start)

            return max_sent_num

        max_nodes = get_batch_stat(data_mb)

        # batching
        minibatch_size = len(data_mb)
        input_len = len(data_mb[0].input_ids)
        id_mb = [d.unique_id for d in data_mb]
        adj_mb = []
        input_ids = torch.zeros(minibatch_size, input_len, dtype=torch.long)
        input_mask = torch.zeros(minibatch_size, input_len, dtype=torch.long)
        segment_ids = torch.zeros(minibatch_size, input_len, dtype=torch.long)
        p_mask = torch.zeros(minibatch_size, input_len, dtype=torch.long)
        for fi, f in enumerate(data_mb):
            input_ids[fi,:] = torch.tensor(f.input_ids, dtype=torch.long)
            input_mask[fi,:] = torch.tensor(f.input_mask, dtype=torch.long)
            segment_ids[fi,:] = torch.tensor(f.segment_ids, dtype=torch.long)
            p_mask[fi,:] = torch.tensor(f.p_mask, dtype=torch.long)
        input_graph_mask = torch.zeros(minibatch_size, max_nodes)
        if self.is_training:
            input_sp_label = torch.zeros(minibatch_size, max_nodes)
            start_pos = torch.zeros(minibatch_size, 1, dtype=torch.long)
            end_pos = torch.zeros(minibatch_size, 1, dtype=torch.long)
            answer_type = torch.zeros(minibatch_size)
        sent_start = -1*torch.ones(minibatch_size, max_nodes, dtype=torch.long)
        sent_end = -1*torch.ones(minibatch_size, max_nodes, dtype=torch.long)

        adj_matrix = gen_adj_matrix(data_mb, max_nodes, \
                wdedge=self.wd_edge, quesedge=self.ques_edge, adedge=self.ad_edge)
        
        # nodes configuration: [cands, docs, mentions, subs]
        for di, d in enumerate(data_mb):
            for si in range(len(d.sent_start)):
                if d.sent_start[si] < input_len:
                    input_graph_mask[di, si] = 1 # sent_mask
                    if self.is_training:
                        input_sp_label[di,si] = d.sp_label[si]
                else:
                    print("Some sentences in sample {} were cut!".format(d.example_index))
            sent_start[di,:len(d.sent_start)] = torch.tensor(d.sent_start)
            sent_end[di,:len(d.sent_start)] = torch.tensor(d.sent_end)
            if self.is_training:
                start_pos[di] = d.start_position
                end_pos[di] = d.end_position
                answer_type[di] = d.answer_type
        if self.is_training:
            return input_ids, input_mask, segment_ids, p_mask, adj_matrix, input_graph_mask, sent_start, sent_end, start_pos, end_pos, input_sp_label, answer_type
        else:
            return input_ids, input_mask, segment_ids, p_mask, adj_matrix, input_graph_mask, sent_start, sent_end

def process_logit(logit, input_data):

    q_count = 0
    output = []
    for idx, entry in enumerate(input_data):
        for qi, q in enumerate(entry['paragraph']['questions']):
            scores = []
            tmp_dict = {}
            tmp_dict['pid'] = entry['id']
            true_logits = list(logit[q_count][:len(q['answers'])])
            for x in true_logits:
                if x >= 0.5:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            tmp_dict['qid'] = "{}".format(qi)
            tmp_dict['scores'] = scores
            q_count += 1
            output.append(tmp_dict)
    
    return output

def evaluate_multirc(logit, oracle_file):

    metrics = {}

    measures = Measures()

    input = json.load(open(oracle_file))
    output = process_logit(logit, input['data'])
    output_map = dict([[a["pid"] + "==" + a["qid"], a["scores"]] for a in output])

    assert len(output_map) == len(output), "You probably have redundancies in your keys"

    [P1, R1, F1m] = measures.per_question_metrics(input["data"], output_map)
    print("Per question measures (i.e. precision-recall per question, then average) ")
    print("\tP: " + str(P1) + " - R: " + str(R1) + " - F1m: " + str(F1m))

    EM0 = measures.exact_match_metrics(input["data"], output_map, 0)
    EM1 = measures.exact_match_metrics(input["data"], output_map, 1)
    print("\tEM0: " + str(EM0))
    print("\tEM1: " + str(EM1))

    [P2, R2, F1a] = measures.per_dataset_metric(input["data"], output_map)

    print("Dataset-wide measures (i.e. precision-recall across all the candidate-answers in the dataset) ")
    print("\tP: " + str(P2) + " - R: " + str(R2) + " - F1a: " + str(F1a))

    metrics['EM'] = EM0
    metrics['F1a'] = F1a

    return metrics