# -*- coding: utf-8 -*-
"""
Created on: 2019/5/27 11:08
@Author: zsfeng
"""
import tensorflow as tf
from tensorflow.contrib import rnn
from model.layer_module import neural_tensor_layer, self_attention, dynamic_routing
from model.base import Base
import numpy as np

"""
代码结构清晰,值得学习 
"""
class InductionGraph(Base):
    def __init__(self, N, K, Q, **kwds):
        """       
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        """
        Base.__init__(self, kwds)
        # c-way k-shot
        self.num_classes = N # 每次batch中有多少个class=5
        self.support_num_per_class = K # k_support, 2
        self.query_num_per_class = Q # k_query, 5
        self.build()

    def forward(self):
        with tf.name_scope("EncoderModule"):
            # embedding_words: [batch=k*c, seq_length, emb_size=embed_size+2*pos_embed_size]
            embedded_words = self.get_embedding()  # (batch=k*c,seq_len,emb_size)

            lstm_fw_cell = rnn.BasicLSTMCell(num_units=self.hidden_size)  # forward direction cell
            lstm_bw_cell = rnn.BasicLSTMCell(num_units=self.hidden_size)  # backward direction cell
            if self.keep_prob is not None:
                lstm_fw_cell = rnn.DropoutWrapper(cell=lstm_fw_cell, output_keep_prob=self.keep_prob)
                lstm_bw_cell = rnn.DropoutWrapper(cell=lstm_bw_cell, output_keep_prob=self.keep_prob)

            # output_fw, output_bw = outputs
            # states_fw, states_bw = states
            # output_fw:[batch, seq_length, hidden_size]
            # state_fw:[batch, hidden_size], hidden state:只记录最后一个hidden_state
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                         cell_bw=lstm_bw_cell,
                                                         inputs=embedded_words,
                                                         dtype=tf.float32)
            # output_rnn: [batch=(k_support+k_query)], seq_length, hidden_size*2]
            output_rnn = tf.concat(outputs, axis=2)  # [k*c,sequence_length,hidden_size*2]
            # encoder: [batch, hidden_size*2]
            # mask_padding:[batch, seq_length]
            mask_padding = tf.cast(self.mask_padding>0, tf.int32) # 原始mask中有1,2,3,0,只有0是padding
            encoder, self.alphas = self_attention(output_rnn, mask_padding)  # (k*c,hidden_size*2)
            # encoder: [batch, hidden_size*2]
            # support_encoder:[batch1=k_support*c, hidden_size*2], support集中的样本数
            # 注意:输入的时候,前面是support集,后面是query集
            support_encoder = tf.slice(input_=encoder, begin=[0, 0],
                                       size=[self.num_classes * self.support_num_per_class, self.hidden_size * 2])
            # query_encoder:[batch1=k_query*c, hidden_size*2], query集中的样本数, 计算query对每类的距离的隐向量
            query_encoder = tf.slice(input_=encoder,
                                     begin=[self.num_classes * self.support_num_per_class, 0],
                                     size=[self.num_classes * self.query_num_per_class, self.hidden_size * 2])
            print("output_rnn:", output_rnn.shape) # [?, 40, 40]
            print("encoder:", encoder.shape) #[?, 40], ? = 10+25 = (2+5)*5 = 35
            print("support_encoder:", support_encoder.shape) # [10, 40]
            print("query_encoder:", query_encoder.shape) # [25, 40]

        # 归纳知识:提取与归纳支持集中每类的表示, 此处没有query集的处理
        with tf.name_scope("InductionModule"):
            # b_IJ:[c=5, k_support=2], 每个类class_i在support的样本j上的权重分数
            b_IJ = tf.constant(np.zeros([self.num_classes, self.support_num_per_class], dtype=np.float32)) # 论文式(5)中的b_s
            print("b_IJ size:", b_IJ.shape) # [5,2]

            # support_encoder:[batch1=k_support*c, hidden_size*2], support集中的样本数
            # class_vector:[c, hidden_size*2], c:class number, 每类的类簇中心向量
            class_vector = dynamic_routing(
                tf.reshape(support_encoder, [self.num_classes, self.support_num_per_class, -1]), # [c, k_support, hidden*2]
                b_IJ)  # (k_support,hidden_size*2)
            print("class vector size:", class_vector.shape) # [c=5, hidden_size*2=40]

        with tf.name_scope("RelationModule"):
            # class_vector:[c, hidden_size*2], c:class number, 各类的隐向量簇中心
            # query_encoder:[batch1=k_query*c, hidden_size*2], query集中的样本数, query在各类上的隐向量表示
            # probs:[batch1=k_query*c=25, c=5], query对各类的相似分数分布,每类下共有k_query个样本
            self.probs = neural_tensor_layer(class_vector, query_encoder) # [k_query*c, c]
            print("probs size:", self.probs.shape)  #[c=5, hidden_size*2=40]

    def build_loss(self):
        with tf.name_scope("loss"):
            # query_label:[batch1=k_query*c]
            # labels_one_hot:[batch1=k_query*c, c=num_classes]
            labels_one_hot = tf.one_hot(self.query_label, self.num_classes, dtype=tf.float32)
            # 回归平方损失
            losses = tf.losses.mean_squared_error(labels=labels_one_hot, predictions=self.probs)

            #l2_losses = tf.add_n( [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda
            # 加上l2惩罚
            l2_losses = []
            for index, v in enumerate(tf.trainable_variables()):
                #print("index:{} var:{}".format(index, v.name))
                if 'bias' not in v.name \
                        and 'word_embedding' not in v.name \
                        and 'pos1_embedding' not in v.name \
                        and 'pos2_embedding' not in v.name:
                    l2_losses.append(v)
                    print("index:{} add l2 loss to var:{}".format(index, v.name))

            self.loss = losses + tf.add_n([tf.nn.l2_loss(v) for v in l2_losses]) * self.l2_lambda
            self.loss = losses
