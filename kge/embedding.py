import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import tensorflow as tf
print("is_gpu: ", tf.test.is_gpu_available())
# part0:重要参数
word_embedding_dim = 50
entity_embedding_dim = 50
kge_method = 'TransE'

# part1:word embedding
from gensim.models import word2vec
w2v_model = word2vec.Word2Vec.load('word_embeddings_' + str(word_embedding_dim) + '.model')

word_to_index = {}
with open("word_to_index.txt", 'r') as file:
    for line in file.readlines():
        # print(line)
        line = line.strip().split('\t')
        word_to_index[line[0]] = line[1]
file.close()

import numpy as np
print('getting word embeddings ...')
embeddings = np.zeros([len(word_to_index) + 1, word_embedding_dim])
for index, word in enumerate(word_to_index.keys()):
    embedding = w2v_model.wv[word] if word in w2v_model.wv.key_to_index else np.zeros(word_embedding_dim)
    embeddings[index + 1] = embedding
print('- saving word embeddings ...')
np.save(('word_embeddings_' + str(word_embedding_dim)), embeddings)

# part2:entity and context embedding
entity_to_index = {}
with open("entity_to_index.txt", 'r') as file:
    for line in file.readlines():
        line = line.strip().split('\t')
        entity_to_index[line[0]] = int(line[1])
file.close()

# 获取邻居实体
entity_to_neighbor = {}
with open("kg.txt", 'r') as file:
    for line in file.readlines():
        line = line.strip().split('\t')
        head = int(line[0])
        tail = int(line[2])
        if head in entity_to_neighbor:
            if tail not in entity_to_neighbor[head]:
                entity_to_neighbor[head].append(tail)
        else:
            entity_to_neighbor[head] = [tail]
        if tail in entity_to_neighbor:
            if head not in entity_to_neighbor[tail]:
                entity_to_neighbor[tail].append(head)
        else:
            entity_to_neighbor[tail] = [head]
file.close()

print('getting entity embeddings ...')
embeddings = np.loadtxt(kge_method + '_entity2vec_' + str(entity_embedding_dim) + '.vec')
entity_embeddings = np.zeros([len(entity_to_index) + 1, entity_embedding_dim])
context_embeddings = np.zeros([len(entity_to_index) + 1, entity_embedding_dim])

for entity, index in entity_to_index.items():
    entity_embeddings[index] = embeddings[index]     # 自己搭建的kg，两个index相同
    if index in entity_to_neighbor:
        context_indices = entity_to_neighbor[index]
        # print(context_indices)    # 保证了相邻实体不重复
        context_embeddings[index] = np.average(embeddings[context_indices], axis=0)

print('- saving entity embeddings ...')
np.save('entity_embeddings_' + kge_method + '_' + str(entity_embedding_dim), entity_embeddings)
print('- saving context embeddings ...')
np.save('context_embeddings_' + kge_method + '_' + str(entity_embedding_dim), entity_embeddings)

