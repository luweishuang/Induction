# -*- coding: utf-8 -*-

import sys
# from util.data_loader import JSONFileDataLoader
from util.data_loader_chinese import JSONFileDataLoader
from model.graph import InductionGraph

model_name = 'induction'
N = 5
K = 5
if len(sys.argv) > 1:
    model_name = sys.argv[1]
if len(sys.argv) > 2:
    N = int(sys.argv[2])
if len(sys.argv) > 3:
    K = int(sys.argv[3])

print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
print("Model: {}".format(model_name))

max_length = 37
# embedding_file = "./data/glove.6B.50d.json"
embedding_file = "./data/sgns.merge.word.json"
# train_intent.json: train_class_num =  53   almost >=30
# val_intent.json:   val_class_num =  81     almost >=6
train_data_loader = JSONFileDataLoader('./data/train_intent.json', embedding_file, max_length=max_length)
val_data_loader = JSONFileDataLoader('./data/val_intent.json', embedding_file, max_length=max_length)


if model_name == 'induction':
    model = InductionGraph(N=N,
                           K=K,
                           Q=5,
                           pred_embed=train_data_loader.word_vec_mat,
                           sequence_length=max_length,      # 这里是一个固定的length,都padding到40
                           hidden_size=20
                           )
    model.train(dataloader=(train_data_loader, val_data_loader),
                model_dir_path="checkpoints/inductionNetwork_test")
