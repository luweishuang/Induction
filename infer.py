# -*- coding: utf-8 -*-

import tensorflow as tf
from util.data_loader_chinese import JSONFileDataLoader


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

    cpkt_meta = './checkpoints/ckpt-5-5/inductionNetwork-12000.meta'
    ckpt = './checkpoints/ckpt-5-5/inductionNetwork-12000'
    with tf.Graph().as_default():
        graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(graph=graph, config=config) as sess:
            saver = tf.train.import_meta_graph(cpkt_meta)
            saver.restore(sess, ckpt)
            print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))

            input_words = graph.get_tensor_by_name('input_words:0')
            encoder = graph.get_tensor_by_name('EncoderModule/self_attn/Squeeze:0')
            # support_encoder.set_shape((N*K, 40))             # set_shape只是设置placeholder的shape
            # query_encoder = graph.get_tensor_by_name('EncoderModule/Slice_1:0')
            # query_encoder.set_shape((N*Q, 40))
            query_label = graph.get_tensor_by_name('query_label:0')
            keep_prob = graph.get_tensor_by_name('keep_probx:0')
            mask_padding = graph.get_tensor_by_name('mask_padding:0')
            accuracy = graph.get_tensor_by_name('accuracy:0')
            loss = graph.get_tensor_by_name('loss/one_hot:0')

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

# Incompatible shapes: [25] vs. [50]         N=10
# Expected size[0] in [0, 5], but got 25     N=3
# Incompatible shapes: [25] vs. [40]         N=8
# Incompatible shapes: [25] vs. [75]         N=15

