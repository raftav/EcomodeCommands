from __future__ import print_function
from __future__ import division

# Avoid printing tensorflow log messages
import os
import tensorflow as tf
import numpy as np
import freeze
import sys
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES']='4'


pb_file='frozen_model_exp1_epoch14.pb'



test_features=sorted(glob.glob('/home/local/IIT/rtavarone/Data/PreProcessEcomodeItalian/DELTAS_TFRECORDS/ANTONIO_FEAT/*.csv'))

#data = np.load('/home/local/IIT/rtavarone/Data/PreProcessEcomode/RecurrentData_NoDeltas/TEST_NUMPY/features_sentence{}.npy'.format(sentence_index),allow_pickle=False)
#data_length = np.array([data.shape[0]])

#data=np.expand_dims(data,axis=0)

#print('data shape = ',data.shape)
#print('data dtype = ',data.dtype)

#print('data len shape = ',data_length.shape)
#print('data len dtype = ',data_length.dtype)

#test_labels=np.load('/home/local/IIT/rtavarone/Data/PreProcessEcomode/RecurrentData_NoDeltas/TEST_NUMPY/labels_sentence{}.npy'.format(sentence_index),allow_pickle=False)
#test_labels = np.expand_dims(test_labels,axis=0)

overall_accuracy=0.0

graph = freeze.load(pb_file)
print('Graph loaded!')

with tf.Session(graph=graph) as sess:

	input_layer = graph.get_tensor_by_name('prefix/inputs/I:0')
	print(input_layer)
	sequence_length = graph.get_tensor_by_name('prefix/inputs/LEN:0')
	print(sequence_length)
	output_layer = graph.get_tensor_by_name('prefix/model/SMO:0')

	labels=tf.placeholder(tf.int64, [1], name='y-input')

	posteriors=tf.nn.softmax(output_layer)
	prediction=tf.argmax(output_layer,axis=2)

	correct = tf.equal(prediction, labels)

	accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")



	for feat_file in test_features:
		data=np.loadtxt(feat_file,delimiter=',',usecols=range(72))
		data_length = np.array([data.shape[0]])
		data=np.expand_dims(data,axis=0)

		test_labels= np.array(0.0)
		test_labels = np.expand_dims(test_labels,axis=0)
		leb, pred , post, acc = sess.run([labels,prediction,posteriors, accuracy], 
						feed_dict={input_layer: data, sequence_length: data_length, labels: test_labels})

		basename_feat=os.path.basename(feat_file)

		print('feature from {}   ----   '.format(basename_feat))
		print('posteriors: ',post)
		print('posteriors sum = ',np.sum(post))
		print('')

		print('label: {} , prediction: {} , accuracy: {}'.format(leb,pred,acc))

		overall_accuracy+=acc
		print('')

print('overall test accuracy = ',overall_accuracy/len(test_features))

