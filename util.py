# -*- coding:utf-8 -*-
#2018-11-6
import jieba

def save_data(datas, path):
    with open(path, 'w', encoding='utf-8')as f:
        for data in datas:
            f.write(data + '\n')

def jieba_comments():
    comments_file = "data/comments.txt"
    f = open(comments_file, 'r')
    comments = " ".join(jieba.cut(f.read()))
    comments = comments.splitlines()
    f.close()
    save_data(comments, "data/comments_jieba.txt")

if __name__ == "__main__":
    jieba_comments()
