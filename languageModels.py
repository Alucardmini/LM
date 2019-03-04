# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '3/4/19'

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]

# word2vec
from gensim.models import Word2Vec
model = Word2Vec(sentences=sentences, min_count=1, size=128)
print(model.wv["cat", "say", "meow"])
print(model.wv["cat", "say", "meow"].shape)

# elmo
# from bilm.training import
from allennlp.modules.elmo import Elmo, batch_to_ids
from bilm.data import BidirectionalLMDataset
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)
character_ids = batch_to_ids(sentences)
embeddings = elmo(character_ids)

print(embeddings['elmo_representations'])




