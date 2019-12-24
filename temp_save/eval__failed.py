# -*- coding: utf-8 -*-

import tensorflow as tf
from util.data_loader_chinese import JSONFileDataLoader
from model.graph import InductionGraph


N = 10     # 受限于 support_encoder 和 query_encoder 的尺寸，N 必须和训练时的大小一致
K = 5
Q = 5
print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))


# train_intent.json: train_class_num =  53   almost >=30
# val_intent.json:   val_class_num =  30     almost > 15
if __name__ == '__main__':
    max_length = 37
    embedding_file = "./data/sgns.merge.word.json"
    val_data_loader = JSONFileDataLoader('./data/val_intent.json', embedding_file, max_length=max_length)
    ckpt = './checkpoints/ckpt-5-5/inductionNetwork-12000'

    model = InductionGraph(N=N,
                           K=K,
                           Q=Q,
                           pred_embed=val_data_loader.word_vec_mat,
                           sequence_length=max_length,
                           hidden_size=20
                           )

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, ckpt)

        input_words = sess.graph.get_tensor_by_name('input_words:0')
        query_label = sess.graph.get_tensor_by_name('query_label:0')
        keep_prob = sess.graph.get_tensor_by_name('keep_probx:0')
        mask_padding = sess.graph.get_tensor_by_name('mask_padding:0')
        accuracy = sess.graph.get_tensor_by_name('accuracy:0')

        iter_right_val, iter_sample_val = 0.0, 0.0
        val_iter = 1000
        for it_val in range(val_iter):
            inputs_val, query_label_val = val_data_loader.next_one_tf(N, K, Q)
            curr_acc_val = sess.run(
                accuracy,
                feed_dict={input_words: inputs_val['word'],
                           query_label: query_label_val,
                           keep_prob: 1,
                           mask_padding: inputs_val['mask']}
            )
            iter_right_val += curr_acc_val
            iter_sample_val += 1
            if it_val % 100 == 0:
                print('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it_val + 1, 100 * iter_right_val / iter_sample_val) + '\r')
        acc_val = iter_right_val / iter_sample_val