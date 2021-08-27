import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import tensorflow as tf
print("is_gpu: ", tf.test.is_gpu_available())
# ;LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64

# part1:读取模型
import kashgari
model = kashgari.utils.load_model('../BERT_CRF/my_bert_crf.h5')

# part2:读取数据
import pandas as pd
data = pd.read_parquet("/mnt/disk2/wanghai/benchmark_adv_table/tx_his/tx_fz/data/part-00035-3b03d0f2-bcd4-463c-9d65-00d461eff8bd-c000.snappy.parquet")

# part3:序列标注
user_id = list(data['userId_output'])
title = list(data['title_output'])   # 发现部分title是None

info_title = zip(user_id, title)
info_title = [two for two in info_title if two[1] != None]    # 去掉空title部分
info_title = sorted(info_title, key=lambda tep: tep[0])     # 根据用户id排个序
user_id, title = zip(*info_title)
user_id = list(user_id)
title = list(title)

entity = []
le = 5000 #len(title)
for i in range(le):
    print(i, '/', le)
    pre = model.predict([[word for word in title[i]]])[0]
    index = 0
    entity.append([])
    while index < len(pre):
        if pre[index] == 'O':
            index = index + 1
            continue
        else:
            tep = index
            index = index + 1
            while index < len(pre) and pre[index] != 'O' and pre[index][2] == pre[index - 1][2]:
                index = index + 1
            entity[i].append(title[i][tep:index])

# part4:保存信息
with open("userid_title_entity.txt", 'w') as file:
    for i in range(le):
        if len(entity[i]) != 0:    # 只保留能够筛选到实体的样本
            entity[i] = '，'.join(entity[i])
            print(entity[i])
            file.write('%s\t%s\t%s\n' % (user_id[i], title[i], entity[i]))
file.close()

