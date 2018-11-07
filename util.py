# -*- coding:utf-8 -*-
#2018-11-6
import jieba
import re

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

def jieba_comments():
    """
    use jieba to cut the comments.txt
    """
    comments_file = "data/comments.txt"
    f = open(comments_file, 'r')
    comments = " ".join(jieba.cut(f.read()))
    comments = comments.splitlines()
    f.close()
    save_data(comments, "data/comments_jieba.txt")

def generate_sentences():
    sentences = read_data("data/comments.txt")
    total_sentences = list()
    for sentence in sentences:
        sentence_tmp = re.findall(r'.{25}', sentence)
        total_sentences.extend(sentence_tmp)
    #print(total_sentences)
    print(len(total_sentences))
    save_data(total_sentences, "data/commits_25.txt")

if __name__ == "__main__":
    generate_sentences()
