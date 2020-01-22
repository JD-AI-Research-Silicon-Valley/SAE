import torch
import glob
import code
import argparse
import csv
from dataset import ExampleDataset, batchify
from dataloader import DocDataset
import json
import pickle
import numpy as np
import torch
import os
import code
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from prepro import get_features
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import pickle
import math
from pytorch_pretrained_bert import (WEIGHTS_NAME, BertConfig,
    BertForSequenceClassification, BertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig, XLMForSequenceClassification,
    XLMTokenizer, XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer)
from pytorch_pretrained_bert import AdamW, WarmupLinearSchedule
from scipy import stats

#files = glob.glob('results/ensemble_large*/pytorch_model.bin')

parser = argparse.ArgumentParser()
parser.add_argument('--dev_name', type=str, default='dev_orig_bert-base-uncased.pkl')
parser.add_argument('--test_name', type=str, default='dev_orig_bert-base-uncased.pkl')
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
parser.add_argument("--bert_model", default='bert-large-uncased', type=str,
                    help='Bert Model to use for tokenization')
parser.add_argument('--output_name', default='',type=str)
args = parser.parse_args()

#with open(args.dev_name, 'rb') as f:
    #eval_features = pickle.load(f)
#with open(args.test_name, 'rb') as f:
    #test_features = pickle.load(f)

#models=[]
#for f in files:
    #print('files', f)
    #models.append(torch.load(f))
##args = models[0]['args']    

model = BertForSequenceClassification.from_pretrained('models/bert-large-uncased-whole-word-masking/',
          num_labels = 2,
          layers = 1,
          weight = 0)

dev_file = args.peng_dev_file if args.use_peng else args.orig_dev_file

tokenizer = BertTokenizer.from_pretrained('models/bert-large-uncased-whole-word-masking/', do_lower_case=args.do_lower_case)
#ddPeval_features = get_features(os.path.join(args.data_dir, dev_file), tokenizer, args.use_mini)
eval_features = get_features(args.dev_name, tokenizer, args.use_mini)
eval_data = ExampleDataset(eval_features, 350)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, num_workers = 4, sampler=eval_sampler, batch_size=20, collate_fn = batchify)

#test_data = ExampleDataset(test_features, 350)
#test_sampler = SequentialSampler(test_data)
#test_dataloader = DataLoader(test_data, num_workers = 20, sampler=test_sampler, batch_size=16, collate_fn = batchify)

files = glob.glob('models/doc_filter/pytorch_model.bin')
def sample_accuracy_and_recall(out, docs):
    num_docs = docs.size(1) 
    #labels=labels.reshape(-1,num_docs,num_docs)
    out=out.reshape(-1, num_docs, num_docs)
    print(out.shape)
    out[out>0.5]=1
    out[out<=0.5]=0
    #truth = np.sum(labels, axis=-1)
    #t=[]
    #for i in range(len(truth)):
        #t.append(np.where(truth[i] != 0)[0].tolist())
    #truth=t
    all_preds = np.sum(out, axis=-1).argsort(axis=-1)
    preds = np.sum(out, axis=-1).argsort(axis=-1)[:, -2:]
    all_correct=0
    total=0
    return all_correct, total, all_preds, None


models=[]
for f in files:
    print('files', f)
    models.append(torch.load(f))
#models=[models[0],models[1],models[2],models[5]]

distributed_models=[]
for i in range(len(models)):
    d={}
    for j, key in enumerate(models[i].keys()):
        d['module.'+key]=models[i][key]
    distributed_models.append(d)
device = torch.device('cuda')
model=model.to(device)
model=torch.nn.DataParallel(model)
all_results=[]

for i in range(len(distributed_models)):
    model.load_state_dict(distributed_models[i])
    print('loaded new model')
    eval_accuracy=0
    eval_recall=0
    nb_eval_examples=0
    nb_eval_steps=0
    model.eval()
    results=[]
    eval_acc, eval_recall, num_examples, truth = 0,0,0,[]
    for batch in eval_dataloader:
        batch = tuple(t.to(device) if type(t) == torch.Tensor else t for t in batch )
        inputs = {
            'input_ids':batch[0],
            'segment_ids':batch[1],
            'input_mask':batch[2],
            'triplet':batch[4],
            'triplet_mask':batch[5],
        }
        with torch.no_grad():
            logits = model(**inputs)
            logits = logits.detach().cpu().numpy()
            #labels = batch[3].squeeze(-1).to('cpu').numpy() 
            labels=None
            tmp_acc, tmp_recall, preds, tmp_truth = sample_accuracy_and_recall(logits, batch[0])
            #eval_acc+=tmp_acc
            #eval_recall+=tmp_recall
            #truth+=tmp_truth
            num_examples+=int(batch[0].size(0))
            #num_docs = int(math.sqrt(labels.shape[-1]))
            #out=logits
            #out=out.reshape(-1,num_docs,num_docs)
            #out[out>0.5]=1
            #out[out<=0.5]=0
            #preds = np.sum(out, axis=-1).argsort(axis=-1)[:, -2:]
            results.append(preds)
    #print(eval_acc, num_examples, eval_acc/num_examples)
    #print(eval_recall, num_examples, eval_recall/(2*num_examples))
    results=np.concatenate(results, axis=0)
    all_results.append(results)

#print('This is the length of all results')
#with open('all_results.pkl','wb') as f:
    #pickle.dump(all_results, f)

all_results = np.stack(all_results, axis=0)
#with open('all_results3.pkl','wb') as f:
    #pickle.dump(all_results, f)
#with open('all_results3.pkl','rb') as f:
    #all_results = pickle.load(f)
#all_results = np.stack(all_results, axis=0)

all_results=all_results[:,:, -2:]

final_results=[]
for i in range(len(all_results[0])):
    preds=[]
    for j in range(len(all_results)):
        preds.append(all_results[j][i])
    preds = np.concatenate(preds, axis=0)
    unique, counts = np.unique(preds, return_counts=True)
    order = np.argsort(counts)
    tmp_result = list(unique[order][-2:])
    final_results.append([int(x) for x in tmp_result])
    
#for i in range(len(all_results[0])):
    #preds = []
    #for j in range(len(all_results)):
        #preds.append(all_results[j][i])
    #preds = np.concatenate(preds, axis=0)
    #preds=preds.reshape(-1,2)
    #preds = stats.mode(preds)[0].tolist()[0] 
    #final_results.append(preds)

#acc=0
#rec=0
#for i, docs in enumerate(final_results):
    #if docs[0] in truth[i] and docs[1] in truth[i]:
        #acc+=1
        #rec+=2
    #elif docs[0] in truth[i] or docs[1] in truth[i]:
        #rec+=1
#print('acc', acc/len(final_results))
#print('recall', rec/(2*len(final_results)))

#with open('truth.json','w') as f:
#    json.dump(truth, f)

with open(args.output_name,'w') as f:
    json.dump(final_results, f)

#print('This is the length of all results')
#with open('all_results3.pkl','wb') as f:
    #pickle.dump(all_results, f)

#code.interact(local=locals())
#exit()
    #logits = logits.detach().cpu().numpy()
    #label_ids = batch[3]
    #label_ids = label_ids.squeeze(-1).to('cpu').numpy()
    #tmp_eval_accuracy, tmp_eval_recall = sample_accuracy_and_recall(logits, label_ids)

    ##eval_loss += tmp_eval_loss.mean().item()
    #eval_accuracy += tmp_eval_accuracy
    #eval_recall += tmp_eval_recall

    #nb_eval_examples += batch[0].size(0)
    #nb_eval_steps += 1

#eval_accuracy = eval_accuracy / nb_eval_examples
#eval_recall = eval_recall / (2*nb_eval_examples)
#result = {  #'loss': loss,
            #'eval_accuracy': eval_accuracy,
            #'eval_recall': eval_recall, 
            ##'global_step': global_step,
            #}

#logger.info("*****EVAL results *****")
#for key in sorted(result.keys()):
    #logger.info("  %s = %s", key, str(result[key]))

