# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '3/4/19'

import tensorflow as tf

import bert.optimization as optimization
import bert.tokenization as tokenization

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]


# word2vec
def word2vec():
    from gensim.models import Word2Vec
    model = Word2Vec(sentences=sentences, min_count=1, size=128)
    print(model.wv["cat", "say", "meow"])
    print(model.wv["cat", "say", "meow"].shape)


# elmo
# from bilm.training import
def elmo_demo():
    from allennlp.modules.elmo import Elmo, batch_to_ids
    # from bilm.data import BidirectionalLMDataset
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    elmo = Elmo(options_file, weight_file, 2, dropout=0)
    character_ids = batch_to_ids(sentences)
    embeddings = elmo(character_ids)
    print(embeddings['elmo_representations'])


def bert_demo():

    # 2	也开不了花呗，就这样了？完事了	真的嘛？就是花呗付款	0

    src_str_1 = '也开不了花呗，就这样了？完事了'
    src_str_2 = '真的嘛？就是花呗付款'

    text_a = tokenization.convert_to_unicode(src_str_1)
    text_b = tokenization.convert_to_unicode(src_str_2)

    v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")
    v2 = tf.Variable(tf.random_normal([2, 3]), name="v2")

    with tf.Session() as sess:
        model = tf.train.import_meta_graph('/home/wuxikun/下载/LM_research/sim_output/model.ckpt-0.meta')
        model.restore(sess, tf.train.latest_checkpoint('/home/wuxikun/下载/LM_research/sim_output/'))

        print(sess.run(tf.global_variables()))


if __name__ == '__main__':
    bert_demo()


