import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn 
import dgl

import sys,time

from coattn import CoAttn_simple, CoAttn
from sort_helper import SeqSortHelper

from scipy.sparse import coo_matrix

# use allennlp implementation to save memory
from allennlp.modules.matrix_attention.linear_matrix_attention import LinearMatrixAttention

MIN_VAL = -99999999

class GCATLayer(nn.Module): # here GCAT is short for "Graph co-attention network"
    def __init__(self, input_dim, rnn_size, dropout):
        super(GCATLayer, self).__init__()
        self.input_dim = input_dim
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.fs = nn.Linear(self.input_dim, self.input_dim) 
        #self.coattn = nn.Linear(3*self.input_dim,1,bias=False)
        self.coattn = LinearMatrixAttention(self.input_dim, self.input_dim, combination='x,y,x*y')
        self.proj = nn.Linear(4*self.input_dim, self.input_dim)
    
    def gen_mask(self, max_len, lengths, device):
        lengths = lengths.type(torch.LongTensor)
        num = lengths.size(0)
        vals = torch.LongTensor(range(max_len)).unsqueeze(0).expand(num, -1)+1 # +1 for masking out sequences with length 0
        mask = torch.gt(vals, lengths.unsqueeze(1).expand(-1, max_len)).to(device)
        return mask
    
    def do_coattn(self, src_feat, src_len, dst_feat, dst_len):
        #query_feat: bs X max_len x feat_dim
        #node_feat: bs X max_len x feat_dim

        MIN_VAL = -1e15

        batch_size, T, fdim = src_feat.size()
        J = dst_feat.size(1)

        src_mask = self.gen_mask(src_feat.size(1), src_len, src_feat.device) # N X T
        dst_mask = self.gen_mask(dst_feat.size(1), dst_len, dst_feat.device) # N X J

        # Make a similarity matrix
        # shape = (batch_size, T, J, fdim)            # (N, T, J, D)
        # src_feat_ex = src_feat.unsqueeze(2)     # (N, T, 1, D)
        # src_feat_ex = src_feat_ex.expand(shape) # (N, T, J, D)
        # dst_feat_ex = dst_feat.unsqueeze(1)         # (N, 1, J, D)
        # dst_feat_ex = dst_feat_ex.expand(shape)     # (N, T, J, D)
        # a_elmwise_mul_b = torch.mul(src_feat_ex, dst_feat_ex) # (N, T, J, D)
        # cat_data = torch.cat((src_feat_ex, dst_feat_ex, a_elmwise_mul_b), 3) # (N, T, J, 3D), [h;u;h◦u]
        # S = self.coattn(cat_data).view(batch_size, T, J) # (N, T, J)
        S = self.coattn(src_feat, dst_feat)


        # dealing with padding
        mask = torch.bmm(src_mask.unsqueeze(2).eq(0).float(), dst_mask.unsqueeze(1).eq(0).float()).eq(0)
        S = S.masked_fill(mask, MIN_VAL)

        # Context2Query
        c2q = torch.bmm(F.softmax(S, dim=-1), dst_feat) # (N, T, D) = bmm( (N, T, J), (N, J, D) )
        # Query2Context
        # b: attention weights on the context
        b = F.softmax(torch.max(S, 2)[0], dim=-1) # (N, T)
        q2c = torch.bmm(b.unsqueeze(1), src_feat) # (N, 1, D) = bmm( (N, 1, T), (N, T, D) )
        q2c = q2c.repeat(1, T, 1) # (N, T, D), tiled T times

        # G: query aware representation of each context word
        G = torch.cat((src_feat, c2q, src_feat.mul(c2q), src_feat.mul(q2c)), 2) # (N, T,4D)

        output = self.proj(G)

        return output

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = self.do_coattn(edges.src['z'].view(-1,self.L, self.D), \
            edges.src['l'], edges.dst['z'].view(-1,self.L, self.D), \
                edges.dst['l'])
        return {'e': z2.view(z2.size(0),-1)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        nb = torch.max(nodes.mailbox['e'], dim=1)[0]
        # equation (4)
        h = nb
        return {'h': h}

    def forward(self, g, h, l):
        """[summary]
        
        Arguments:
            g {DGLGraph} -- Input DGL graph
            h {h} -- input node sequence features
            l {l} -- input node sequence lengths
        
        Returns:
            [type] -- [description]
        """
        self.g = g
        self.bsN,self.L,self.D = h.size()
        # equation (1)
        z = self.fs(h).view(self.bsN,-1) # (BXN) X (L X D)
        self.g.ndata['z'] = z # node features
        self.g.ndata['l'] = l # node lengths
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        node_feat = self.g.ndata.pop('h').view(self.bsN,self.L,-1)
        return node_feat

class seqGNN_DGL_BiDAF(nn.Module):
    def __init__(self, input_dim, rnn_size, dropout = 0.1, num_hop=1, gcn_num_rel=2, multirel = True, share=True, gating=True):
        super(seqGNN_DGL_BiDAF, self).__init__()

        self.rnn_size = int(rnn_size)
        self.input_dim = int(input_dim)
        self.dropout = dropout
        self.num_hop = int(num_hop)
        self.gcn_num_rel = int(gcn_num_rel)
        self.sfm = nn.Softmax(-1)
        self.multirel = multirel
        if not self.multirel:
            self.gcn_num_rel = 1
        
        self.share = share
        self.gating = gating

        if share:
            for i in range(gcn_num_rel):
                setattr(self, "gcat{}".format(i+1), GCATLayer(self.input_dim, self.rnn_size, dropout=self.dropout))
            # gating
            if gating:
                self.fa = nn.Sequential(nn.Linear(4*self.rnn_size+self.input_dim, self.input_dim), nn.Tanh(), 
                    nn.Dropout(dropout, inplace=False), nn.Linear(self.input_dim, 1), nn.Tanh())
        else:
            for i in range(num_hop):
                for j in range(gcn_num_rel):
                    setattr(self, "gcat{}_{}".format(i+1,j+1), GCATLayer(self.input_dim, self.rnn_size, dropout=self.dropout))
                # gating
                if gating:
                    setattr(self, "fa{}".format(i+1), nn.Sequential(nn.Linear(4*self.rnn_size+self.input_dim, self.input_dim), nn.Tanh(), 
                        nn.Dropout(dropout, inplace=False), nn.Linear(self.input_dim, 1), nn.Tanh()))
    
    def forward(self, input, input_len, input_mask, adj):
        
        bs,max_nodes,sent_len,feat_dim = input.size()
        input_len = input_len.view(bs*max_nodes)
        cur_input = input.clone().view(bs*max_nodes,sent_len,-1) # max_len X bs X dim

        # prepare graphs
        graphs = [[] for _ in range(self.gcn_num_rel)]
        for r in range(self.gcn_num_rel):
            for i in range(bs):
                g = dgl.DGLGraph()
                g.add_nodes(max_nodes)
                a = coo_matrix(adj[i,r,:,:].cpu().numpy())
                g.from_scipy_sparse_matrix(a)
                g.add_edges(g.nodes(), g.nodes())
                graphs[r].append(g)
            graphs[r] = dgl.batch(graphs[r])
        
        for i in range(self.num_hop):

            # integrate neighbor information
            s_time = time.time()
            nb_output = []
            for j in range(self.gcn_num_rel):
                if self.share:
                    nb_output.append(getattr(self, "gcat{}".format(j+1))(graphs[j],cur_input,input_len))
                else:
                    nb_output.append(getattr(self, "gcat{}_{}".format(i+1, j+1))(graphs[j],cur_input,input_len))
            
            #print("DGL computation finished in {} seconds!".format(time.time()-s_time))
            
            update = torch.sum(torch.stack(nb_output,dim=1),dim=1) / self.gcn_num_rel

            if self.gating:

                # get gate values
                if self.share:
                    gate_tmp = self.fa(torch.cat((update, cur_input), -1)).view(bs,max_nodes,sent_len)
                else:
                    gate_tmp = getattr(self,"fa{}".format(i+1))(torch.cat((update, cur_input), -1)).view(bs,max_nodes,sent_len) 
                gate_tmp = torch.where(gate_tmp == 0.0, -9e15*torch.ones_like(gate_tmp), gate_tmp)
                gate = torch.sigmoid(gate_tmp)* (input_mask.unsqueeze(-1))  # bs x max_node x node_dim
                gate = gate.view(bs*max_nodes,sent_len,-1)

                # apply gate values
                cur_input = gate * torch.tanh(update) + (1 - gate) * cur_input  # (bs* max_node) X max_len X node_dim
            
            else:
                cur_input = update
    
        return cur_input.view(bs,max_nodes,sent_len,-1)