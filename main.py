import sys, subprocess

def main(input_file):

    # step 1: get predicted gold documents
    subprocess.call("CUDA_VISIBLE_DEVICES=0 python -u docfilter.py --do_lower_case \
                    --dev_name {} --output_name output/pred_gold_idx.json".format(input_file), shell=True)
    
    # step 2: data preparation and NER extraction
    subprocess.call("python -u prepare_pred_gold.py output/pred_gold_idx.json \
                    {} output/pred_gold_doc.json".format(input_file), shell=True) 
    
    # step 3: get model output
    subprocess.call("CUDA_VISIBLE_DEVICES=0 python -u run_hotpotqa_roberta.py \
        --sp_from_span --hop 3 --sent_sum_way attn --span_loss_weight 0.3 \
        --wdedge --quesedge --adedge \
        --model_type roberta \
        --model_name_or_path roberta-large \
        --version_2_with_negative \
        --do_eval \
        --train_file hotpot_train_v1.1.json --train_ner_file bert_position_ner_hotpot_train_v1.1.json \
        --predict_file output/pred_gold_doc.json --predict_ner_file output/pred_gold_ner.json \
        --learning_rate 2e-5 \
        --weight_decay 0.0 \
        --warmup_steps 0.06 \
        --num_train_epochs 3 \
        --max_seq_length 512 \
        --doc_stride 128 \
        --output_dir models/qa_model/ \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=2 \
        --gradient_accumulation_steps 1", shell=True)

    # step 4: combine span and supporting sentence prediction
    subprocess.call("python combine_pred.py output/predictions_ans.json output/predictions_sp.json", shell=True)


if __name__ == "__main__":
    input_file = sys.argv[1]
    main(input_file)