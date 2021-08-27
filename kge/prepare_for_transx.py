import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import tensorflow as tf
print("is_gpu: ", tf.test.is_gpu_available())

# part1:triple2id.txt
with open("kg.txt", 'r') as file_in, open("triple2id.txt", 'w') as file_out:
    lines = file_in.readlines()
    file_out.write('%d\n' % (len(lines)))
    for line in lines:
        line = line.strip().split('\t')
        file_out.write('%s\t%s\t%s\n' % (line[0], line[2], line[1]))
file_out.close()
file_in.close()

# part2:entity2id.txt
with open("entity_to_index.txt", 'r') as file_in, open("entity2id.txt", 'w') as file_out:
    lines = file_in.readlines()
    file_out.write('%d\n' % (len(lines) + 1))
    # 加入（0，0）保证index取值正确
    file_out.write('%s\t%s\n' % ('0', '0'))
    for line in lines:
        line = line.strip().split('\t')
        print(line)
        file_out.write('%s\t%s\n' % (line[1], line[1]))
file_out.close()
file_in.close()

# part3:relation2id.txt
with open("user_to_index.txt", 'r') as file_in, open("relation2id.txt", 'w') as file_out:
    lines = file_in.readlines()
    file_out.write('%d\n' % (len(lines) + 1))
    file_out.write('%s\t%s\n' % ('0', '0'))
    for line in lines:
        line = line.strip().split('\t')
        file_out.write('%s\t%s\n' % (line[1], line[1]))
file_out.close()
file_in.close()
