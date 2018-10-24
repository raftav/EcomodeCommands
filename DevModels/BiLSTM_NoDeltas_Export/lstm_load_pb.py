from __future__ import print_function
from __future__ import division

# Avoid printing tensorflow log messages
import os
import tensorflow as tf
import numpy as np
import freeze
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


ExpNum = 1
restore_epoch=118

pb_file='frozen_model_exp{}_epoch{}.pb'.format(ExpNum,restore_epoch)

sentence_index=sys.argv[1]
data = np.load('/home/local/IIT/rtavarone/Data/PreProcessEcomode/RecurrentData_NoDeltas/TEST_NUMPY/features_sentence{}.npy'.format(sentence_index),allow_pickle=False)
data_length = np.array([data.shape[0]])

data=np.expand_dims(data,axis=0)

print('data shape = ',data.shape)
print('data dtype = ',data.dtype)

#print('data len shape = ',data_length.shape)
#print('data len dtype = ',data_length.dtype)

test_labels=np.load('/home/local/IIT/rtavarone/Data/PreProcessEcomode/RecurrentData_NoDeltas/TEST_NUMPY/labels_sentence{}.npy'.format(sentence_index),allow_pickle=False)
test_labels = np.expand_dims(test_labels,axis=0)

graph = freeze.load(pb_file)
print('Graph loaded!')

with tf.Session(graph=graph) as sess:

	input_layer = graph.get_tensor_by_name('prefix/inputs/I:0')
	print(input_layer)
	sequence_length = graph.get_tensor_by_name('prefix/inputs/LEN:0')
	print(sequence_length)
	output_layer = graph.get_tensor_by_name('prefix/model/SMO:0')

	labels=tf.placeholder(tf.int64, [1], name='y-input')

	prediction=tf.argmax(output_layer, axis=2)

	correct = tf.equal(prediction, labels)

	accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")

	leb, pred , acc = sess.run([labels,prediction,accuracy], feed_dict={input_layer: data, sequence_length: data_length, labels: test_labels})
	print('label: {} , prediction: {} , accuracy: {}'.format(leb,pred,acc))



