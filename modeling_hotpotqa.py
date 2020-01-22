from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
from pytorch_transformers.modeling_roberta import RobertaModel
from pytorch_transformers.modeling_utils import PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits
from pytorch_transformers.configuration_roberta import RobertaConfig
from pytorch_transformers.file_utils import add_start_docstrings


ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}


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

    def __init__(self, config, num_answer_type=3, num_hop = 3, num_rel = 2, no_gnn=False, gsn=False, edp=0.0, span_from_sp = False, sp_from_span = False, xlnet_spanloss=False, sent_with_cls=False):
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
        self.xlnet_spanloss = xlnet_spanloss
        self.sent_with_cls = sent_with_cls

        # if not self.no_gnn:
        #     self.sp_graph = gcnLayer(config.hidden_size, config.hidden_size, num_hop=num_hop, gcn_num_rel=num_rel,edp=edp)

        if not self.gsn:
            self.sp_graph = gcnLayer(config.hidden_size, config.hidden_size, num_hop=num_hop, gcn_num_rel=num_rel,edp=edp)

        self.hidden_size = int(config.hidden_size/2)

        self.sent_selfatt = nn.Sequential(nn.Linear(config.hidden_size, self.hidden_size), GeLU(), 
                        nn.Dropout(self.dropout), nn.Linear(self.hidden_size, 1))

        if self.xlnet_spanloss:
            self.start_logits = PoolerStartLogits(config)
            self.end_logits = PoolerEndLogits(config)
        else:
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
                start_positions=None, end_positions=None, sp_label=None, all_answer_type=None, sent_sum_way='attn', span_loss_weight = 1.0):

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
        if self.xlnet_spanloss:
            start_logits = self.start_logits(sequence_output, p_mask=p_mask)
            # if start_positions.dim() > 1:
            #     start_positions = start_positions.squeeze(-1)
            if start_positions is not None:
                end_logits = self.end_logits(sequence_output, start_positions=start_positions, p_mask=p_mask)
                #end_logits = self.end_logits(sequence_output, start_states=sequence_output, p_mask=p_mask)
            else:
                n_top = 20
                bsz, slen, hsz = sequence_output.size()
                start_log_probs = F.softmax(start_logits, dim=-1) # shape (bsz, slen)

                start_top_log_probs, start_top_index = torch.topk(start_log_probs, n_top, dim=-1) # shape (bsz, start_n_top)
                start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz) # shape (bsz, start_n_top, hsz)
                start_states = torch.gather(sequence_output, -2, start_top_index_exp) # shape (bsz, start_n_top, hsz)
                start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1) # shape (bsz, slen, start_n_top, hsz)

                hidden_states_expanded = sequence_output.unsqueeze(2).expand_as(start_states) # shape (bsz, slen, start_n_top, hsz)
                p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
                end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
                end_logits = end_logits.mean(-1)
        else:
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

        feat_dim = cls_output.size(-1)

        # sentence extraction
        per_sent_len = sent_end - sent_start
        max_sent_len = torch.max(sent_end - sent_start)
        if self.sent_with_cls:
            per_sent_len += 1
            max_sent_len += 1
        # print("Maximum sent length is {}".format(max_sent_len))
        sent_output = torch.zeros(bs, max_nodes, max_sent_len, feat_dim).to(input_ids.device)
        span_logits = start_logits + end_logits
        sent_span_logits = -9e15*torch.ones(bs,max_nodes,max_sent_len).to(input_ids.device)
        for i in range(bs):
            for j in range(max_nodes):
                if sent_end[i,j] <= sent_len:
                    if sent_start[i,j] != -1 and sent_end[i,j] != -1:
                        if not self.sent_with_cls:
                            sent_output[i,j,:(sent_end[i,j]-sent_start[i,j]),:] = sequence_output[i,sent_start[i,j]:sent_end[i,j],:]
                            sent_span_logits[i,j,:(sent_end[i,j]-sent_start[i,j])] = span_logits[i,sent_start[i,j]:sent_end[i,j]]
                        else:
                            sent_output[i,j,1:(sent_end[i,j]-sent_start[i,j])+1,:] = sequence_output[i,sent_start[i,j]:sent_end[i,j],:]
                            sent_output[i,j,0,:] = cls_output[i]
                            sent_span_logits[i,j,1:(sent_end[i,j]-sent_start[i,j])+1] = span_logits[i,sent_start[i,j]:sent_end[i,j]]
                            sent_span_logits[i,j,0] = -9e15 
                else:
                    if sent_start[i,j] < sent_len:
                        if not self.sent_with_cls:
                            sent_output[i,j,:(sent_len-sent_start[i,j]),:] = sequence_output[i,sent_start[i,j]:sent_len,:] 
                            sent_span_logits[i,j,:(sent_len-sent_start[i,j])] = span_logits[i,sent_start[i,j]:sent_len]
                        else:
                            sent_output[i,j,1:(sent_len-sent_start[i,j])+1,:] = sequence_output[i,sent_start[i,j]:sent_len,:]
                            sent_output[i,j,0,:] = cls_output[i] 
                            sent_span_logits[i,j,1:(sent_len-sent_start[i,j])+1] = span_logits[i,sent_start[i,j]:sent_len]
                            sent_span_logits[i,j,0] = -9e15
        
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