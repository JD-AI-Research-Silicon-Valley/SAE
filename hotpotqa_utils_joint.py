import torch
import numpy as np
import json, sys, re, string
import collections
from collections import Counter
from collections import OrderedDict


def get_sp_pred(pred_sp_idx, data):
    """get the prediction of supporting facts in original format
    
    Arguments:
        pred_sp_idx {[type]} -- [description]
        data {[type]} -- [description]
    """
    pred = []
    for p in pred_sp_idx:
        if p < len(data):
            pred.append([data[p].doc_title[0], data[p].sent_id])

    return pred

def process_logit(batch_index, batch_logits, predict_features, predict_examples, max_answer_length):
    """get predictions for each sample in the batch
    
    Arguments:
        batch_index {[type]} -- [description]
        batch_logits {[type]} -- 0: supporting facts logits, 1: answer span logits, 2: answer type logits 3: gold doc logits
        batch_size {[type]} -- [description]
        predict_file {[type]} -- [description]
    """
    
    sp_logits_np = torch.sigmoid(batch_logits[0]).detach().cpu().numpy()
    ans_type_logits_np = batch_logits[1].detach().cpu().numpy()

    batch_index = batch_index.numpy().tolist()

    sp_pred, span_pred, ans_type_pred = [], [], []

    for idx, data in enumerate(batch_index):

        # supporting facts prediction
        pred_sp_idx = [ x[0] for x in enumerate(sp_logits_np[idx,:].tolist()) if x[1] > 0.5 ]
        print(pred_sp_idx)
        if len(pred_sp_idx) != 0:
            sp_pred.append(get_sp_pred(pred_sp_idx, predict_examples[data]))
        else:
            sp_pred.append([])

        # answer type prediction, for debug purpose
        ans_type_pred.append(np.argmax(ans_type_logits_np[idx,:]))

        # answer span prediction
        if ans_type_pred[-1] == 0:
           span_pred.append("no")
        elif ans_type_pred[-1] == 1:
           span_pred.append("yes")
        else:
           span_pred.append("")
    
    return sp_pred, span_pred, ans_type_pred


# def evaluate(eval_file, answer_dict):
#     f1 = exact_match = total = 0
#     for key, value in enumerate(answer_dict):
#         total += 1
#         ground_truths = eval_file[key]["answer"]
#         prediction = value
#         cur_EM = exact_match_score(prediction, ground_truths)
#         cur_f1, _, _ = f1_score(prediction, ground_truths)
#         exact_match += cur_EM
#         f1 += cur_f1

#     exact_match = 100.0 * exact_match / total
#     f1 = 100.0 * f1 / total

#     return {'exact_match': exact_match, 'f1': f1}

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def write_prediction(sp_preds, answer_preds, orig_data, predict_file, output_dir):
    """write predictions to json file
    
    Arguments:
        sp_preds {[type]} -- [description]
        answer_preds {[type]} -- [description]
        orig_data {[type]} -- [description]
        predict_file {[type]} -- [description]
        output_dir {[type]} -- [description]
    """
    if len(answer_preds) == 0:
        answer_preds = ["place_holder"] * len(orig_data)
    all_pred = {}
    all_pred['answer'] = OrderedDict()
    all_pred['sp'] = OrderedDict()
    for idx, data in enumerate(orig_data):
        all_pred['answer'][data['_id']] = answer_preds[idx]
        all_pred['sp'][data['_id']] = sp_preds[idx]

    with open(output_dir, 'w') as fid:
        json.dump(all_pred, fid)

