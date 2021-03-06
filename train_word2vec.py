#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by Ross on 18-4-12
import config
import joblib
import logging
import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors

LOG_FILE = 'W2V.log'
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=5)
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger('word2vec_training')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 语料库
corpus = 'data/total_commit_to_vec.txt'
# output_path = os.path.join(config.get_data_path(), '10G_16dim_train.pkl')
logging.info('Initialized')


def train(dim):
    sentences = LineSentence(corpus)
    logger.info('Training...')

    w2v = Word2Vec(sentences, size=dim)

    logger.info('Finished training')

    logger.info('Saving model...')

    output_path = os.path.join("data", 'dim_Douban_with_trash.pkl')
    joblib.dump(w2v, output_path)
    logger.info('Model saved')


def test(output_path):
    w2v = joblib.load(output_path)
    # print(w2v['知乎'])
    # print(w2v['学习'])
    # print(Word2Vec(w2v).most_similar(positive=['知乎', '微博'], negative=['酒店']))
    print(w2v[u'操你妈'])
    print('pass')


if __name__ == '__main__':
    #train(32)
    test("data/dim_Douban_with_trash.pkl")
