import nltk
import string
import numpy as np
import pandas as pd
from multiprocessing import Pool
from pandas import DataFrame
import spacy
from spacy.lang.en import English
from time import time
from nltk.tokenize import TweetTokenizer

nlp = spacy.load('en_core_web_sm')
spacy_tokenizer = English().Defaults.create_tokenizer(nlp)
tweet_tokenizer = TweetTokenizer()


##============================================================
def parallelize_dataframe(dataframe: DataFrame, func, npartitioons = 8, processes = 8):
    """
    :param dataframe:
    :param func:
    :param npartitioons:
    :param processes:
    :return:
    """
    npartitioons = min(dataframe.shape[0], npartitioons) #for example: Avoid divide "8 into 10 pieces"
    df_split = np.array_split(dataframe, npartitioons, axis= 0)
    pool = Pool(processes=processes)
    dataframe = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return dataframe



def ner_noun_tokenizer_sentences(sentences: list):
    """
    Name entity and noun phrase extraction from a tokenized sentence
    :param sentence_tokens:
    :param query_entity:
    :return:
    """
    name_entity_list = []
    noun_phrases_list = []
    tokens_list = []

    # nlp.tokenizer = TweetTokenizer().tokenize
    for sentence in sentences:
        sent_nlp = nlp(sentence)
        entities = [ent.text for ent in sent_nlp.ents]
        noun_phrases = [noun.text for noun in sent_nlp.noun_chunks if noun.text not in entities]
        tokens = [token.text for token in sent_nlp]

        name_entity_list.append(entities)
        noun_phrases_list.append(noun_phrases)
        tokens_list.append(tokens)

    return [name_entity_list, noun_phrases_list, tokens_list]

def ner_noun_tokenizer_sentence(sentence):
    """
    Name entity and noun phrase extraction from a tokenized sentence
    :param sentence_tokens:
    :param query_entity:
    :return:
    """
    # nlp.tokenizer = TweetTokenizer()
    sent_nlp = nlp(sentence)
    entities = [ent.text for ent in sent_nlp.ents]
    noun_phrases = [noun.text for noun in sent_nlp.noun_chunks if noun.text not in entities]
    tokens = [token.text for token in sent_nlp]

    return [entities, noun_phrases, tokens]

def tokenizer_sentence(sentence, type = 'tweet'):
    if type == 'tweet':
        sent_nlp = tweet_tokenizer.tokenize(sentence)
        tokens = [token for token in sent_nlp]
    else:
        sent_nlp = spacy_tokenizer(sentence)
        tokens = [token.text for token in sent_nlp]
    return tokens