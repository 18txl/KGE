import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import tensorflow as tf
print("is_gpu: ", tf.test.is_gpu_available())

# part1:读取实体
user_index = []
entity_index = []
with open("user_title_word_entity_index.txt", 'r') as file:
    for line in file.readlines():
        line = line.strip().split('\t')
        user_index.append(line[0])
        # 删除用于补位的'0'
        entity_index_now = line[3].split('，')
        while True:
            if '0' in entity_index_now:
                entity_index_now.remove('0')
                continue
            else:
                break
        entity_index.append(entity_index_now)
file.close()

# part2:构建图谱
user_entity = {}
le = len(user_index)
for i in range(le):
    if user_index[i] in user_entity:
        user_entity[user_index[i]] = list(set(user_entity[user_index[i]]) | set(entity_index[i]))   # 求列表并集
    else:
        user_entity[user_index[i]] = entity_index[i]

# 相同user点击过的实体相连
kg = []
for user, entity in user_entity.items():
    for j in range(len(entity)):
        k = 0
        while k < len(entity):
            if k != j:
                kg.append((entity[j], user, entity[k]))
            k = k + 1

# part3:保存图谱
with open("kg.txt", 'w') as file:
    for i in range(len(kg)):
        file.write('%s\t%s\t%s\n' % (kg[i][0], kg[i][1], kg[i][2]))
file.close()

