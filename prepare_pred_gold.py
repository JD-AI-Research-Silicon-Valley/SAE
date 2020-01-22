import sys, json, pickle, subprocess

def write_prediction(preds,data_file, output_file):
    with open(data_file) as fid:
        eval_data = json.load(fid)

    assert(len(preds) == len(eval_data)) # check the numbers of samples match
    from collections import OrderedDict

    filtered_eval = [{} for _ in range(len(eval_data))]
    for k,ex in enumerate(eval_data):
        filtered_eval[k]['question'] = ex['question']
        filtered_eval[k]['_id'] = ex['_id']
        filtered_eval[k]['context'] = []
        if len(ex['context']) == 2:
            print(ex['_id'])
            filtered_eval[k]['context'] = ex['context'] 
        else:
            for j,para in enumerate(ex['context']):
                if j in preds[k]:
                    filtered_eval[k]['context'].append(para)
    with open(output_file, 'w') as fid:
        json.dump(filtered_eval, fid)

    subprocess.call("python -u NERExtractorTest.py {} {}".format(output_file, output_file.replace('doc', 'ner')), shell=True)

if __name__ == '__main__':
    preds_file = sys.argv[1]
    data_file = sys.argv[2]
    #data_file = 'hotpot_dev_distractor_v1.json'
    output_file = sys.argv[3]
    #with open(preds_file, "rb") as reader:
    #    preds = pickle.load(reader)
    with open(preds_file) as reader:
        preds = json.load(reader)
    write_prediction(preds, data_file, output_file)
