# -*- coding:utf-8 -*-
#2018-11-6
import jieba
import re
import numpy as np 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def save_data(datas, path):
    """
    save the datas to the path
    param:
        datas -> the data that want to be saved
        path -> the location of the saved data
    """
    with open(path, 'w', encoding='utf-8')as f:
        for data in datas:
            f.write(data + '\n')

def read_data(filename):
    """
    read the data
    param:
        filename -> the location and the filename of the data that want to read
    """
    with open(filename, 'r', encoding='utf-8')as f:
        data = f.read().splitlines()
    return data

def jieba_comments(load_filename, save_filename):
    """
    use jieba to cut the comments.txt
    """
    comments_file = load_filename
    f = open(comments_file, 'r')
    jieba.load_userdict("data/trashwords.txt")
    comments = " ".join(jieba.cut(f.read()))
    comments = comments.splitlines()
    f.close()
    save_data(comments, save_filename)

def generate_sentences():
    sentences = read_data("data/comments.txt")
    total_sentences = list()
    for sentence in sentences:
        sentence_tmp = re.findall(r'.{25}', sentence)
        total_sentences.extend(sentence_tmp)
    #print(total_sentences)
    print(len(total_sentences))
    save_data(total_sentences, "data/commits_25.txt")

def get_trashword():
    trashes = read_data("data/trashwords.txt")
    commits = read_data("data/commits_25.txt")
    print(len(trashes))
    train_split_out, test_split_out = train_test_split(commits, test_size=0.1, random_state=33)
    print(len(train_split_out), len(test_split_out))
    
    #training smaples
    train_normal, trash = get_which_trash(train_split_out)
    commit_trash_train = add_trash(trash, train_split_out, trashes)
    commit_normal = get_normal(train_normal, train_split_out)
    commit_train = list()
    commit_train = commit_normal+commit_trash_train
    print(type(commit_train))
    print(len(commit_train))
    save_data(commit_train, "data/commit_train.txt")

    #testing samples
    test_normal, test_trash = get_which_trash(test_split_out)
    commit_trash_test = add_trash(test_trash, test_split_out, trashes)
    commit_normal_test = get_normal(test_normal, test_split_out)
    commit_test = list()
    commit_test = commit_normal_test+commit_trash_test
    save_data(commit_test, "data/commit_test.txt")

def get_normal(normal_index, datas):
    sentences_origin = list()
    for index in normal_index:
        sentences_origin.append(datas[index]+"\t0")
    return sentences_origin

def add_trash(trash_index, commits, trashes):
    added_sentences = list()
    for index in trash_index:
        random_split = random(25)
        random_trash = random(27)
        str_tmp = str(commits[index][0:random_split]) + trashes[random_trash] + str(commits[index][random_split:]+"\t1")
        added_sentences.append(str_tmp)
    return added_sentences

def random(number):
    return np.random.randint(0, number)

def get_which_trash(datas):
    kf = KFold(n_splits=2, shuffle=True, random_state=33)
    #train_index_return, trash_index_return = None
    for train_index, trash_index in kf.split(datas):
        #print("TRAIN:", train_index, len(train_index), "TEST:", trash_index, len(trash_index))
        train_index_return, trash_index_return = train_index, trash_index
        break
    return train_index_return, trash_index_return

def word2vector():
   vec = joblib.load("data/dim_Douban_with_trash.pkl")
   print(type(vec["操你妈"]))

if __name__ == "__main__":
    word2vector()
