import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import json

'''
    Do the stupid way first: multiply the number of examples by ~5 and toss it into our dataset
    How all of this should work
        -> have two Tensor Dataset
            ->one is for correct Documents * 5?
            ->one is for incorrect Documents
        -> have a sampler that chooses between them equally?
        -> therefore, batches should have an approximately equal amount of pos/neg samples
'''
class DocDataset(Dataset):
    def __init__(self, *dataset):
        self.dataset = self._make_dataset(*dataset)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def _make_dataset(self, *dataset):
        p, n = [], []
        for i, label in enumerate(dataset[-1]):
            print(label)
            print(dataset[i+1])
            exit()
            if label==1:
                p.append(label)
            else:
                n.append(label)
        w=len(n)/len(p)
        p=p*(w+1)
        return torch.TensorDataset(p+n)

class SplitDocDataset(Dataset):
    def __init__(self, dataset):
        self.pos_dataset, self.neg_dataset = self._split_dataset(dataset)

    def __len__(self):
        return len(self.pos_dataset) + len(self.neg_dataset)

    def __getitem__(self, index, value):
        if value==1:
            return self.pos[index]
        else:
            return self.neg[index]

    def _split_dataset(dataset):
        p, n = [], []
        for doc in dataset:
            if doc.label == 1:
                p.append(doc)
            else:
                n.append(doc)
        return torch.TensorDataset(p), torch.TensorDataset(n)

#Probably Not Needed
class DocOverSampler(Sampler):
    def __init__(self, dataset):
        self.dataset=dataset
        
    def __iter__(self):
        pass

    def __len__(self):
        return len(self.dataset)

class OverSampler(Sampler):
    def __init__(self, dataset, sampler, batch_size=1):
        self.dataset=dataset
        self.sampler = sampler
        self.batch_size = batch_size

    def __len__(self):
        return len(self.sampler)
        
