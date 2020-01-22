import numpy as np
import math, random, json
import time, sys, subprocess, os
import glob, pickle

from torch.utils.data import Dataset, DataLoader

#allennlp
# from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
# from allennlp.modules.elmo import Elmo, batch_to_ids

# parallel processing
import pickle


class whDataset(Dataset):
    '''
    Dataset class to read in scp and label files
    '''
    def __init__(self, batch_folder):

        self.batch_folder = batch_folder

        self.batch_files = glob.glob(batch_folder + '/*.pkl')

        self.num_batch = len(self.batch_files)


    def load_pkl(self, filename):

        with open(filename, 'rb') as bfid:
            st = time.time()
            batch = pickle.load(bfid)
            print('Batch {} loaded in {} seconds!'.format(os.path.basename(filename), time.time()-st))
        return batch

    def __getitem__(self, index):
        return self.load_pkl(self.batch_folder + '/bacth{}.pkl'.format(index))

    def __len__(self):
        return self.num_batch

def _collate_fn(batch):

    return batch[0]

class whDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(whDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

class SeqSortHelper(object):
    def __init__(self):
        self.orders = None

    #sort input based on sequence lengths
    def sort_input(self, input, input_len):    #TODO
        input_data, lengths = input, input_len

        #input_data: seqL X batch_size X dim
        l1, perm_idx = lengths.sort(descending=True)
        l2, perm_idx_back = perm_idx.sort()
        perm_idx = perm_idx.to(input_data.device)
        perm_idx_back = perm_idx_back.to(input_data.device)

        input_data = input_data.index_select(1, perm_idx).contiguous() 
        input_len = lengths.index_select(0, perm_idx).contiguous()
        self.orders = perm_idx_back
        return input_data, input_len, perm_idx_back

    #restore the original order of batch
    #input: (seqL X) batch_size X dim
    def restore_order_input(self, input, perm_idx_back):
        #orders = self.orders.to(input.device)
        input = input.index_select(1, perm_idx_back)
        return input

    def reset(self):
        self.orders = None


# data_preparation test
if __name__ == '__main__':
    start_pos = int(sys.argv[1])*4384
    train_data = whDataset("/mnt/cephfs2/asr/users/ming.tu/work/NLP/datasets/qangaroo_v1.1/wikihop/train.json", batch_size=8)
    train_data.save_all_batches("/opt/cephfs1/asr/users/mingtu/work/NLP/entityGCN/batch_data_sgcn8/train", start_pos)