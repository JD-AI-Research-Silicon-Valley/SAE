from Core.IOUtils import loadJSonAsDataFrame, save_dataframe2json, save_dataframe2json2
from time import time

from pandas import DataFrame
import gc, sys
import pandas as pd
from DataUtils.nerExtractor import parallelize_dataframe, ner_noun_tokenizer_sentence, ner_noun_tokenizer_sentences, tokenizer_sentence
from pytorch_pretrained_bert.tokenization import BertTokenizer, PRETRAINED_VOCAB_ARCHIVE_MAP


bert_tokenizer = BertTokenizer.from_pretrained("models/bert-base-uncased-vocab.txt")

def context_extractor(doc_list: list):
    context_names = []
    context_doc_sentence_num = []
    context_doc_num = len(doc_list)
    for doc in doc_list:
        context_names.append(doc[0])
        context_doc_sentence_num.append(len(doc[1]))
    return pd.Series([context_doc_num, context_doc_sentence_num, context_names])

def ner_tokenize_row(row):
    question = row['question']
    [ner, noun, tokens] = ner_noun_tokenizer_sentence(question)
    context_list = row['context']
    context_ner_list = []
    context_tokens_list = []
    context_noun_list = []

    for idx, context in enumerate(context_list):
        title = context[0]
        [tit_ner_list, tit_noun_list, tit_tokens_list] = ner_noun_tokenizer_sentence(title)
        sentences = context[1]
        [sent_ner_list, sent_noun_list, sent_tokens_list] = ner_noun_tokenizer_sentences(sentences)

        context_tokens_list.append([tit_tokens_list, sent_tokens_list])
        context_ner_list.append([tit_ner_list, sent_ner_list])
        context_noun_list.append([tit_noun_list, sent_noun_list])
    return pd.Series([ner, noun, tokens, context_ner_list, context_noun_list, context_tokens_list])

def bert_tokenize_row(row):
    question = row['question']
    question_tokens = bert_tokenizer.tokenize(question)
    context_list = row['context']
    context_tokens_list = []

    for idx, context in enumerate(context_list):
        title = context[0]
        tit_tokens_list = bert_tokenizer.tokenize(title)
        sentences = context[1]
        sentence_tokens = [bert_tokenizer.tokenize(sentence) for sentence in sentences]

        context_tokens_list.append([tit_tokens_list, sentence_tokens])

    return pd.Series([question_tokens, context_tokens_list])

def ner_noun_on_context(data_frame: DataFrame):
    start = time()
    data_frame[['question_ner',
                'question_noun',
                'question_token',
                'context_ner',
                'context_noun',
                'context_token',
                ]] = data_frame.apply(lambda row: ner_tokenize_row(row), axis= 1)
    print('Ner extracting on {} tokens {} seconds'.format(data_frame.shape[0], time() - start))
    return data_frame

def ner_noun_on_context_par(data_frame: DataFrame):
    data_frame = parallelize_dataframe(data_frame, ner_noun_on_context, npartitioons=2048, processes=8)
    return data_frame

def bert_tokenizer_df(data_frame: DataFrame):
    start = time()
    data_frame[['question_token_bert',
                'context_token_bert']] = data_frame.apply(lambda row: bert_tokenize_row(row), axis= 1)
    print('Bert tokening {} rows in {} seconds'.format(data_frame.shape[0], time() - start))
    return data_frame

def bert_ner_noun_position(ners, nouns, tokens):
    ner_positions = bert_position_extraction(ners, tokens)
    noun_positions = bert_position_extraction(nouns, tokens)
    return [ner_positions, noun_positions]

def bert_position_extraction(target_list, source_tokens):
    t_idx = 0
    s_idx = 0
    positions = []
    while t_idx < len(target_list) and s_idx < len(source_tokens):
        target = target_list[t_idx]
        target_tokens = bert_tokenizer.tokenize(target)
        target_len = len(target_tokens)
        while s_idx < len(source_tokens) and s_idx + target_len <= len(source_tokens):
            sub_tokens = source_tokens[s_idx:(s_idx+target_len)]
            if check_equal_sequence(sub_tokens, target_tokens):
                # positions.append({'text': target, 'text_token': target_tokens, 'start': s_idx, 'end': s_idx + target_len})
                positions.append([target, target_tokens, s_idx, s_idx + target_len])
                s_idx = s_idx + target_len
                break
            else:
                s_idx = s_idx + 1
                continue
        t_idx = t_idx + 1
    return positions


def bert_ner_noun_position_row(row):
    question_token, question_ner, question_nouns = row['question_token_bert'], row['question_ner'], row['question_noun']
    question_ner_positions, question_noun_position = bert_ner_noun_position(question_ner, question_nouns, question_token)

    context_token, context_ner, context_nouns = row['context_token_bert'], row['context_ner'], row['context_noun']
    context_ner_positions = []
    context_noun_position = []
    for doc_idx, cont_token in enumerate(context_token):
        title_token, title_ner, title_noun = cont_token[0], context_ner[doc_idx][0], context_nouns[doc_idx][0]
        title_ner_position, title_noun_position = bert_ner_noun_position(title_ner, title_noun, title_token)
        context_ner_positions.append(title_ner_position)
        context_noun_position.append(title_noun_position)
        sentence_token, sentence_ner, sentence_noun = cont_token[1], context_ner[doc_idx][1], context_nouns[doc_idx][1]
        ner_positions = []
        noun_positions = []
        for sent_idx, sentence in enumerate(sentence_token):
            sent_i, ner_i, noun_i = sentence, sentence_ner[sent_idx], sentence_noun[sent_idx]
            ner_position_i, noun_position_i = bert_ner_noun_position(ner_i, noun_i, sent_i)
            ner_positions.append(ner_position_i)
            noun_positions.append(noun_position_i)
        context_ner_positions.append(ner_positions)
        context_noun_position.append(noun_positions)


    return pd.Series([question_ner_positions, question_noun_position, context_ner_positions, context_noun_position])

def check_equal_sequence(sup_token_list, entity_token_list):
    """
    Whether two sequence lists are identical
    :param sup_token_list:
    :param entity_token_list:
    :return:
    """
    import operator as op
    for i in range(0, len(entity_token_list)):
        if op.eq(sup_token_list[i].lower(), entity_token_list[i].lower()):
            continue
        else:
            return False
    return True


def position_map_token(tokens, bert_tokens):
    position_map = []
    token_idx = 0
    bert_idx = 0
    while (token_idx < len(tokens) and bert_idx < len(bert_tokens)):
        token: str = tokens(token_idx)
        bert_token: str = bert_tokens[bert_idx]
        if token == bert_token:
            position_map.append((bert_idx, bert_idx + 1))
            token_idx = token_idx + 1
            bert_idx = bert_idx + 1
        elif token.startswith(bert_token):
            temp_idx = bert_idx + 1
            temp_str = bert_token
            while(temp_idx < len(bert_tokens) and bert_tokens[temp_idx].startswith('#')):
                temp_str = temp_str + bert_tokens[temp_idx]
                temp_idx = temp_idx + 1
            token_idx = token_idx + 1
            position_map.append((bert_idx, temp_idx - 1))
            bert_idx = temp_idx
    return position_map


def bert_tokenizer_par(data_frame: DataFrame):
    data_frame = parallelize_dataframe(data_frame, bert_tokenizer_df, npartitioons=1024, processes=8)
    return data_frame

def bert_ner_position_df(data_frame: DataFrame):
    start = time()
    data_frame[['question_ner_pos', 'question_noun_pos', 'context_ner_pos', 'context_noun_pos']] = data_frame.apply(lambda row: bert_ner_noun_position_row(row), axis=1)
    print('Position extracting on {} tokens {} seconds'.format(data_frame.shape[0], time() - start))
    return data_frame

def bert_ner_position_par(data_frame: DataFrame):
    data_frame = parallelize_dataframe(data_frame, bert_ner_position_df, npartitioons=512, processes=8)
    return data_frame
#===================================================================================================

def bert_merge(token_file_name, bert_token_name):
    df = loadJSonAsDataFrame(token_file_name)
    bert_df = loadJSonAsDataFrame(bert_token_name)
    start = time()
    merge_df = pd.merge(df[['_id', 'question_ner',
                'question_noun', 'context_ner', 'context_noun']], bert_df[['_id', 'question_token_bert','context_token_bert']], on = '_id')
    print('Merging {} takes {}'.format(merge_df.shape[0], time() - start))
    return merge_df

if __name__ == '__main__':
    
    dev_name = sys.argv[1]
    bert_token_position_dev_name = sys.argv[2]
    ner_token_dev_name = "output/ner_token_hotpot_dev_distractor_v1.json"
    bert_token_dev_name = "output/bert_token_hotpot_dev_distractor_v1.json"

    ner_token_names = ['_id','question_ner',
                'question_noun',
                'question_token',
                'context_ner',
                'context_noun',
                'context_token']
    bert_tokens = ['_id','question_token_bert',
                'context_token_bert']
    pos_names = ['_id', 'question_ner_pos', 'question_noun_pos', 'context_ner_pos', 'context_noun_pos']

    data_frame = loadJSonAsDataFrame(dev_name)
    ##Step 1: NER Noun extractor
    start = time()
    ner_data_frame = ner_noun_on_context_par(data_frame)
    ner_data_frame = ner_data_frame[ner_token_names]
    print('Step 1Runtime = {}'.format(time() - start))
    save_dataframe2json2(ner_data_frame, ner_token_dev_name)

    ##Step 2: BERT tokenizing
    start = time()
    bert_token_df = bert_tokenizer_par(data_frame)
    print('Step 2 Runtime = {}'.format(time() - start))
    save_dataframe2json2(bert_token_df, bert_token_dev_name)


    ##Step 3: Combine two data frames together and position extraction
    start = time()
    bert_token_ner_df = bert_merge(ner_token_dev_name, bert_token_dev_name)
    positon_df = bert_ner_position_par(bert_token_ner_df)
    print('Step 3 Runtime = {}'.format(time() - start))
    save_dataframe2json2(positon_df, bert_token_position_dev_name)






