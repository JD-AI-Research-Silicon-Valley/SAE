import torch
import torch.nn as nn
import math, sys


MIN_VAL = -99999999


        
class CoAttn(nn.Module):
    def __init__(self, input_dim, attn_dim=0, adapt_scale = False):
        super(CoAttn, self).__init__()
        if adapt_scale:
            self.scale = nn.Parameter(torch.FloatTensor([1./math.sqrt(input_dim)]))
        else:
            self.scale = 1./math.sqrt(input_dim)

        self.sfm = nn.Softmax(-1)

    def check_model(self, level):
        print("%sCoAttn:scale %.3f"%(level, self.scale))

    def forward(self, query, node, maskq, maskn):
        #query: seqL1 X batch_size X dim
        #maskq: batch_size X seqL1
        #node: seqL2 X batch_size X dim, batch_size can be real batch_size multipled by number of docs/candidates
        #maskn: batch_size X seqL2

        # challenge: batch size of query and node are different

        seqL1, bs1, dim = query.size()
        seqL2, bs2, dim = node.size()
        num_node = int(bs2/bs1)

        query = query.unsqueeze(-2).expand(seqL1,bs1,num_node,dim).contiguous().view(seqL1,bs1*num_node,dim)
        maskq = maskq.unsqueeze(-2).expand(bs1,num_node,maskq.size(-1)).contiguous().view(bs1*num_node,maskq.size(-1))

        query2 = query.transpose(1,0)
        node2 = node.transpose(1,0)

        prod = torch.bmm(node2, query2.transpose(1,2))*self.scale #batch_size X seqL2 X seqL1
        mask = torch.bmm(maskn.unsqueeze(2).eq(0).float(), maskq.unsqueeze(1).eq(0).float()).eq(0)
        prod = prod.masked_fill(mask, MIN_VAL)

        #\hat(input1)
        wts1 = self.sfm(prod) #batch_size X seqL2 X seqL1
        wts1 = wts1.masked_fill(mask,0)
        output_ss = torch.bmm(wts1, query.transpose(1,0)).transpose(0,1) #seqL2 X batch_size X dim


        #\hat(input2)
        wts2 = self.sfm(prod.transpose(-1,-2)) #batch_size X seqL1 X seqL2
        wts2 = wts2.masked_fill(mask.transpose(-1,-2), 0)
        output_sq = torch.bmm(wts2, node.transpose(1,0)).transpose(0,1) #seqL1X batch_size X dim

        cs_input = torch.bmm(wts1, output_sq.transpose(1,0)).transpose(0,1)

        return output_ss, output_sq, cs_input

class CoAttn_simple(nn.Module):
    def __init__(self, input_dim, attn_dim=0, adapt_scale = False):
        super(CoAttn_simple, self).__init__()
        if adapt_scale:
            self.scale = nn.Parameter(torch.FloatTensor([1./math.sqrt(input_dim)]))
        else:
            self.scale = 1./math.sqrt(input_dim)

        self.sfm = nn.Softmax(-1)

    def check_model(self, level):
        print("%sCoAttn:scale %.3f"%(level, self.scale))

    def forward(self, query, node, maskq, maskn, coattn_weight):
        #query: seqL1 X batch_size X dim
        #maskq: batch_size X seqL1
        #node: seqL2 X batch_size X dim, batch_size can be real batch_size multipled by number of docs/candidates
        #maskn: batch_size X seqL2

        # challenge: batch size of query and node are different

        seqL1, bs1, dim = query.size()
        seqL2, bs2, dim = node.size()
        num_node = int(bs2/bs1)

        query = query.unsqueeze(-2).expand(seqL1,bs1,num_node,dim).contiguous().view(seqL1,bs1*num_node,dim)
        maskq = maskq.unsqueeze(-2).expand(bs1,num_node,maskq.size(-1)).contiguous().view(bs1*num_node,maskq.size(-1))

        query2 = query.transpose(1,0)
        node2 = node.transpose(1,0)

        prod_tmp = coattn_weight(node2)
        prod = torch.bmm(prod_tmp, query2.transpose(1,2))*self.scale #batch_size X seqL2 X seqL1
        mask = torch.bmm(maskn.unsqueeze(2).eq(0).float(), maskq.unsqueeze(1).eq(0).float()).eq(0)
        prod = prod.masked_fill(mask, MIN_VAL)

        #\hat(input2)
        wts2 = self.sfm(prod.transpose(-1,-2)) #batch_size X seqL1 X seqL2
        wts2 = wts2.masked_fill(mask.transpose(-1,-2), 0)
        output_sq = torch.bmm(wts2, node.transpose(1,0)).transpose(0,1) #seqL1X batch_size X dim

        return output_sq