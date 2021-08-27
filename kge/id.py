import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import tensorflow as tf
print("is_gpu: ", tf.test.is_gpu_available())
# part0:重要参数
max_title_length = 10
word_embedding_dim = 50

# part1:读取实体
users = []
titles = []
entities = []
words = []
words_no = ['，', '—', '、', '？', '“', '”', '！', ':', '/', '（', '）', '《', '》', '-', '【', '】', ',', '：', '~', '·',
            '.', '!', '#', '。', '「', '」', '%', '…', ' ', '　', '  ', '～', '｜', '@', '°', '+', '[', ']', '；', '?',
            '_', '→','↘', '’', '&', '＃', '㎡', '丨', '•', '"', '．', '－', '％', '＋', '✨', '℃', '(', ')', '\u200d']
with open("userid_title_entity.txt", 'r') as file:
    for line in file.readlines():
        line = line.strip().split('\t')
        users.append(line[0])
        titles.append(line[1])
        entities.append([entity for entity in line[2].split('，') if entity != ''])
        words.append([word for word in line[1] if word not in words_no])
file.close()

# part2:user to index
index = 1
uesr_to_index = {}
for user in users:
    if user not in uesr_to_index:
        uesr_to_index[user] = index
        index = index + 1

# part3:entity to index
index = 1
entity_to_index = {}
for entity in entities:
    for entity_single in entity:
        if entity_single not in entity_to_index:
             entity_to_index[entity_single] = index
             index = index + 1

# part4:word to index
index = 1
word_to_index = {}
for word in words:
    for word_single in word:
        if word_single not in word_to_index:
            word_to_index[word_single] = index
            index = index + 1

# part5:title to index
index = 1
title_to_index = {}
for title in titles:
    if title not in title_to_index:
        title_to_index[title] = index
        index = index + 1

# part6:保存对应关系
with open("user_to_index.txt", 'w') as file:
    for tep, index in uesr_to_index.items():
        file.write('%s\t%s\n' % (tep, str(index)))
file.close()

with open("entity_to_index.txt", 'w') as file:
    for tep, index in entity_to_index.items():
        print(tep, index)
        file.write('%s\t%s\n' % (tep, str(index)))
file.close()

with open("word_to_index.txt", 'w') as file:
    for tep, index in word_to_index.items():
        print(tep, index)
        file.write('%s\t%s\n' % (tep, str(index)))
file.close()

with open("title_to_index.txt", 'w') as file:
    for tep, index in title_to_index.items():
        print(tep, index)
        file.write('%s\t%s\n' % (tep, str(index)))
file.close()

# part7:name to index
user_index = []
title_index = []
word_index = []
entity_index = []

le = len(users)
for i in range(le):
    user_index.append(uesr_to_index[users[i]])
    title_index.append(title_to_index[titles[i]])
    word_single_index = ['0'] * max_title_length
    entity_single_index = ['0'] * max_title_length
    for num, word in enumerate(words[i]):
        if num < max_title_length:
            word_single_index[num] = str(word_to_index[word])
    for num, entity in enumerate(entities[i]):
        if num < max_title_length:
            entity_single_index[num] = str(entity_to_index[entity])
    word_single_index = '，'.join(word_single_index)
    entity_single_index = '，'.join(entity_single_index)
    word_index.append(word_single_index)
    entity_index.append(entity_single_index)

with open("user_title_word_entity_index.txt", 'w') as file:
    for i in range(le):
        file.write('%s\t%s\t%s\t%s\n' % (str(user_index[i]), str(title_index[i]), str(word_index[i]), str(entity_index[i])))
file.close()

# part8:train.txt 和 test.txt
# 点击新闻数大于5的才选取
user_qualified = []
click_news_cnt = 9
test_cnt = int(click_news_cnt/3)
for user in set(users):
    if users.count(user) > click_news_cnt:
        user_qualified.append(user)

with open("train.txt", 'w') as train, open("test.txt", 'w') as test:
    le = len(users)
    i = 0
    while i < le:
        if users[i] in user_qualified:
            # 该用户的编号，起始序号和终止序号
            user = user_index[i]
            begin = i
            while user_index[i] == user:
                i = i + 1
            end = i
            # 前k条写入测试集，其余放入训练集，label为1
            title_now = []
            i = begin
            while i < begin + test_cnt:
                test.write('%s\t%s\t%s\t%s\n' % (str(user_index[i]), word_index[i], entity_index[i], '1'))
                title_now.append(titles[begin])
                i = i + 1
            while i < end:
                train.write('%s\t%s\t%s\t%s\n' % (str(user_index[i]), word_index[i], entity_index[i], '1'))
                title_now.append(titles[i])
                i = i + 1
            # 在title中随机选择k个未点击的放入测试集，随机选择2k个未点击的放入训练集，label为0
            wl = random.sample(range(1, le), 2*click_news_cnt)
            for t in wl:
                if titles[t] in title_now:
                    wl.remove(t)
            if len(wl) > test_cnt:
                for n in range(test_cnt):
                    test.write('%s\t%s\t%s\t%s\n' % (str(user), word_index[wl[n]], entity_index[wl[n]], '0'))
            if len(wl) > click_news_cnt + 1:
                for n in range(test_cnt, click_news_cnt):
                    train.write('%s\t%s\t%s\t%s\n' % (str(user), word_index[wl[n]], entity_index[wl[n]], '0'))
            else:
                for n in range(test_cnt, len(wl)):
                    train.write('%s\t%s\t%s\t%s\n' % (str(user), word_index[wl[n]], entity_index[wl[n]], '0'))
        else:
            i = i + 1

# part9:word embeddings model
from gensim.models import word2vec
w2v_model = word2vec.Word2Vec(titles, vector_size=word_embedding_dim, min_count=1, workers=16)
w2v_model.save('word_embeddings_' + str(word_embedding_dim) + '.model')
