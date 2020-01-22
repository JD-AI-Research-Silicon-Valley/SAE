import sys, subprocess
import json
from collections import OrderedDict

def clear_punc(text):
    puncs = [",","-","'",".","?","!"]
    new_text = ""
    for i in range(0,len(text)):
        if text[i] == " " and (text[i-1] in puncs or text[i+1] in puncs):
            continue
        else:
            new_text += text[i]
    return new_text

def main(span_pred_file, sp_pred_file):

    with open(span_pred_file) as fid:
        span_pred = json.load(fid)
    
    with open(sp_pred_file) as fid:
        sp_pred = json.load(fid)
    
    comb_pred = {}
    comb_pred['answer'] = OrderedDict()
    comb_pred['sp'] = OrderedDict()
    for key, val in span_pred.items():
        if key not in sp_pred['answer']:
            comb_pred['answer'][key] = ""
        else:
            if sp_pred['answer'][key] == 'yes':
                comb_pred['answer'][key] = 'yes'
            elif sp_pred['answer'][key] == 'no':
                comb_pred['answer'][key] = 'no'
            else:
                comb_pred['answer'][key] = val
    
    for key, val in sp_pred['sp'].items():
            comb_pred['sp'][key] = sp_pred['sp'][key]
    
    with open('pred.json','w') as fid:
        json.dump(comb_pred, fid)

if __name__ == "__main__":
    span_pred_file = sys.argv[1]
    sp_pred_file = sys.argv[2]
    main(span_pred_file, sp_pred_file)