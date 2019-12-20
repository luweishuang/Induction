# -*- coding: utf-8 -*-
"""
Created on: 2019/5/27 14:29
@Author: zsfeng
"""
import tensorflow as tf

def neural_tensor_layer(class_vector, query_encoder, out_size=100):
    """neural tensor layer (NTN)"""
    # class_vector: [C, H]
    # query_encoder: [K*C, H]
    C, H = class_vector.shape
    # print("class_vector shape:", class_vector.shape)
    # print("query_encoder shape:", query_encoder.shape)
    M = tf.get_variable("M", [H, H, out_size], dtype=tf.float32,
                        initializer=tf.keras.initializers.glorot_normal())
    mid_pro = []
    for slice in range(out_size):
        # class_vector:[C, H]
        # M[:,:,slice]: [H, H],注意是2维矩阵
        # class_m:[C, H]
        class_m = tf.matmul(class_vector, M[:, :, slice]) # [5, 40]
        #print("class_m:", class_m, "m slice:", M[:,:,slice])
        # class_m:[C, H]
        # query_encoder: [K*C, H]
        # slice_inter:[C=5, K*C=25=5*5]
        slice_inter = tf.matmul(class_m, query_encoder, transpose_b=True)  # (C,Q)
        #print("slice_inter:", slice_inter)
        # list of [C, K*C]
        mid_pro.append(slice_inter)

    # tensor_bi_product:[C*out_size, K*C]
    tensor_bi_product = tf.concat(mid_pro, axis=0)  # (C*out_size, K*C)
    print("tensor_bi_product shape:{}".format(tensor_bi_product.shape))
    V = tf.nn.relu(tf.transpose(tensor_bi_product)) # [K*C, C*out_size]
    W = tf.get_variable("w", [C * out_size, C], dtype=tf.float32,
                        initializer=tf.keras.initializers.glorot_normal())
    b = tf.get_variable("b", [C], dtype=tf.float32,
                        initializer=tf.keras.initializers.glorot_normal())
    # probs:[batch=k_query*c, c], 每类下有k_query, 每个query都预测属于某个类的概率
    probs = tf.nn.sigmoid(tf.matmul(V, W) + b)  # [batch=K*C, C],不知为何不用softmax
    return probs

def self_attention(inputs, mask):
    """
    a = softmax(Wa2*tanh(Wa1*H^T))
    e =sum_{t=1}{at*ht}

    a:[1,T]
    Wa2:[1,da]
    Wa1:[da, 2u]
    H^T:[2u, T]
    """
    # inputs: [batch=(k_support+k_query)], seq_length, hidden_size]
    _, sequence_length, hidden_size = inputs.shape
    with tf.variable_scope('self_attn'):
        # project到某一个维度之后,再与另一个w作点乘得到权重，然后与原x作加权求和
        # inputs:[batch, seq_length, hidden_size]
        # x_proj:[batch, seq_length, hidden_size]
        x_proj = tf.layers.Dense(units=hidden_size)(inputs) # W_a1:[hidden_size, hidden_size],应该是最后一维上相乘
        print("x_proj shape:", x_proj.shape) # [?, 40, 40]
        x_proj = tf.nn.tanh(x_proj)
        # u_w:[hidden_size, 1]
        u_w = tf.get_variable('W_a2',
                              shape=[hidden_size, 1],
                              dtype=tf.float32,
                              initializer=tf.keras.initializers.glorot_normal())
        # x_proj:[batch, seq_length, hidden_size]
        # u_w:[hidden_size, 1]
        # x:[batch, seq_length, 1]
        x = tf.tensordot(a=x_proj, b=u_w, axes=1) # tensordot:在x_proj的倒数第axes=1维上以及u_w的第axes=1维上矩阵乘积
        print("x shape:", x.shape) # [?, 37, 1]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.

        # mask_adder:[batch, seq_length, 1]
        mask_adder = (1.0 - tf.cast(tf.expand_dims(mask, -1), tf.float32)) * (-2**31)

        # alphas:[batch, seq_length, 1]
        alphas = tf.nn.softmax(x + mask_adder, axis=1) # 在各时间步上计算softmax
        print("alphas shape", alphas.shape)
        # inputs_trans: [batch, hidden_size, seq_length]
        inputs_trans = tf.transpose(a=inputs, perm=[0, 2, 1])
        # inputs_trans: [batch, hidden_size, seq_length],batch中各时间步的向量
        # alphas:[batch, seq_length, 1], 各时间步的系数
        # output:[batch, hidden_size, 1], 各时间步加求和后的向量
        output = tf.matmul(inputs_trans, alphas) # 类似于batch_matmul
        # output:[batch, hidden_size]
        output = tf.squeeze(output, axis=-1)
        # output:[batch, hidden_size]
        return output, alphas


def dynamic_routing(input, b_IJ, iter_routing=3):
    # 归纳网络:对支持集中不同的类进行归纳
    ''' The routing algorithm.'''
    """
    input:[c, k_support, hidden]
    b_IJ:[c, k_support], 式(5)中的bs
   
    e'^s_ij = squash(Ws* e^s_ij + bs)
    Ws:[2u,2u] 
    e^s_ij:[2u,1], class i sample j
    bs:[2u,1]
    
    支持集中的所有向量共享同一个w_s以及bias
    """
    C, K, H = input.shape
    W = tf.get_variable(name='W_s',
                        shape=[H, H],
                        dtype=tf.float32,
                        initializer=tf.keras.initializers.glorot_normal())

    #b = tf.get_variable("Ws_bias", [C], dtype=tf.float32, initializer=tf.keras.initializers.zeros())

    for r_iter in range(iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # b_IJ:[c, k_support]
            # d_I:[c, k_support, 1]
            d_I = tf.nn.softmax(tf.reshape(b_IJ, shape=[C, K, 1]), axis=1) #论文式(7), 对于每个类选择一个比较好的样本d_I
            # for all samples j = 1, ..., K in class i:
            # input reshape:[c* k_support, hidden]
            # W:[hidden, hidden]
            # e_IJ:[c, k_support, hidden]
            e_IJ = tf.reshape(tf.matmul(tf.reshape(input, [-1, H]), W), shape=[C, K, -1])  # (C,K,H), 论文式(5)
            # d_I:[c, k_support, 1]
            # e_IJ:[c, k_support, hidden], multiply为点乘
            # c_I:[c, 1, hidden], 每个类下各样本的加权平均和
            c_I = tf.reduce_sum(tf.multiply(d_I, e_IJ), axis=1, keepdims=True)  # (C,1,H), 论文式(8)
            # c_I:[c, hidden], 每个类的类簇中心
            c_I = tf.reshape(c_I, [C, -1])  # (C,H)
            c_I = squash(c_I)  # (C,H), squash在axis=1维度上归一化
            # e_IJ:[c, k_support, hidden]
            # c_I reshape:[c, hidden, 1], matmul中的两个3d矩阵的 后两维必须match,以适合矩阵相乘,类似于 batch_matmul
            # c_produce_e: [c, k_support, 1]
            c_produce_e = tf.matmul(e_IJ, tf.reshape(c_I, [C, H, 1]))  # (C,K,1)

            # for all samples j = 1, ..., K in class i:
            # b_IJ:[c, k_support]
            b_IJ += tf.reshape(c_produce_e, [C, K]) # 每类与每个样本的耦合系数,论文中式(10)

    # c_I:[c, hidden]
    return c_I


def squash(vector):
    '''Squashing function corresponding to Eq. 1
        vector:[C, H]
        f(x) = x^2/(1+x^2)* x/||x||
    '''
    # vector:[c, hidden]
    vec_squared_norm = tf.reduce_sum(tf.square(vector), axis=1, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-9)
    vec_squashed = scalar_factor * vector  # element-wise
    return vec_squashed


if __name__ == "__main__":
    import numpy as np

    inputs = np.random.random((24, 5, 10))  # (3*3+3*5,seq_len,lstm_hidden_size*2)
    mask = np.ones((24, 5))
    mask[0][4:] = 0
    mask[1][3:] = 0
    mask[2][2:] = 0
    # print (inputs)
    inputs = tf.constant(inputs, dtype=tf.float32)
    encoder, att_mask = self_attention(inputs, mask)  # (k*c,lstm_hidden_size*2)

    support_encoder = tf.slice(encoder, [0, 0], [9, 10])
    query_encoder = tf.slice(encoder, [9, 0], [15, 10])

    support_encoder = tf.reshape(support_encoder, [3, 3, -1])
    b_IJ = tf.constant(np.zeros([3, 3], dtype=np.float32))
    class_vector = dynamic_routing(support_encoder, b_IJ)
    inter = neural_tensor_layer(class_vector, query_encoder, out_size=10)

    # test accuracy
    query_label = [0, 1, 2] * 5
    print(query_label)
    predict = tf.argmax(name="predictions", input=inter, axis=1)
    correct_prediction = tf.equal(tf.cast(predict, tf.int32), query_label)
    accuracy = tf.reduce_mean(name="accuracy", input_tensor=tf.cast(correct_prediction, tf.float32))
    labels_one_hot = tf.one_hot(query_label, 3, dtype=tf.float32)

    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        print(encoder.eval())
        print(query_encoder.eval())
        print(inter.eval())
        print(predict.eval())
        print(correct_prediction.eval())
        print(accuracy.eval())
        print("att_mask:", att_mask.eval())
