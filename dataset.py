import torch
from torch.utils.data import Dataset
import code

#CLS = 101
#SEP = 102
class ExampleDataset(Dataset):
    def __init__(self, input_features, max_seq_length=512):
        self.question = input_features[0] 
        self.docs = input_features[1] 
        self.offsets = input_features[2] 
        self.labels = input_features[3] 
        self.titles = input_features[4] 
        self.max_seq_length = max_seq_length 

    def __len__(self):
        return len(self.question)

    #Get Labels -> / ranking
    def __getitem__(self, idx):
        all_input_ids=[]
        all_segment_ids=[]
        all_input_mask=[]
        for docs in self.docs[idx]:
            input_ids = [101]+self.question[idx]+[102]
            segment_ids = [0]*len(input_ids) + [1]*len(sum(docs,[]))
            if len(segment_ids)>self.max_seq_length:
                segment_ids=segment_ids[:self.max_seq_length]
            else:
                segment_ids+=[0]*(self.max_seq_length-len(segment_ids))
            input_ids += sum(docs,[])

            input_ids = input_ids[:self.max_seq_length-1]
            input_ids.append(102)
            input_ids += [0]*(self.max_seq_length-len(input_ids))
            input_mask = [1]*len(input_ids)+[0]*(self.max_seq_length-len(input_ids))

            all_input_ids.append(input_ids)
            all_segment_ids.append(segment_ids)
            all_input_mask.append(input_mask)

        input_ids = all_input_ids
        input_mask = all_input_mask
        segment_ids = all_segment_ids
        labels=[]
        triplet=[]
        for i in range(len(self.docs[idx])):
            for j in range(len(self.docs[idx])):
                if self.labels[idx] !=[]:
                    doc1 = self.labels[idx][i]
                    doc2 = self.labels[idx][j]
                    if doc1>doc2:
                        labels.append(1)
                    else:
                        labels.append(0) 
                triplet.append([i,j])
        input_ids = torch.tensor(input_ids) #shape should be 10x512
        segment_ids = torch.tensor(segment_ids)
        input_mask = torch.tensor(input_mask)
        labels = torch.tensor(labels)
        triplet = torch.tensor(triplet)
        true_label = self.labels[idx]+[0]*(10-len(self.labels[idx]))
        #print('triplet', triplet)
        return input_ids, segment_ids, input_mask, labels, triplet, torch.tensor(true_label)

def batchify(batch):

    max_len = len(batch)
    max_docs = max([len(ex[0]) for ex in batch])
    max_pairs = max([len(ex[4]) for ex in batch])
    max_seq_len = max([batch[0][1].size(-1) for ex in batch])
    #print('Maximum SEQ LENGTH', max_seq_len)
    input_ids=torch.zeros(max_len, max_docs, max_seq_len).long()
    segment_ids=torch.zeros(max_len, max_docs, max_seq_len).long()
    input_mask=torch.zeros(max_len, max_docs, max_seq_len).long()
    labels=torch.zeros(max_len, max_pairs)
    triplet=torch.zeros(max_len, max_pairs, 2).long()
    triplet_mask=torch.zeros(max_len, max_pairs).float() 
    true_labels = [ex[-1] for ex in batch]
    true_labels = torch.stack(true_labels, dim=0)
    for i, ex in enumerate(batch):
        input_ids[i,:ex[0].size(0),:ex[0].size(-1)].copy_(ex[0])
        segment_ids[i, :ex[1].size(0), :ex[1].size(1)].copy_(ex[1])
        input_mask[i, :ex[2].size(0), :ex[2].size(1)].copy_(ex[2])
        labels[i,:len(ex[3])].copy_(ex[3])
        triplet[i,:len(ex[4]), :].copy_(ex[4])
        triplet_mask[i,:len(ex[4])].fill_(1)
    return input_ids, segment_ids, input_mask, labels, triplet, triplet_mask,  true_labels
