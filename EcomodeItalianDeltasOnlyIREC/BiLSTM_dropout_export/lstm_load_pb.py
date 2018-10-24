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


pb_file='frozen_model_exp1_epoch34.pb'



#test_features=sorted(glob.glob('/home/local/IIT/rtavarone/Data/PreProcessEcomodeItalian/DELTAS_TFRECORDS/TEST_FEAT_IREC_NUMPY/features_*.npy'))
#test_labels=sorted(glob.glob('/home/local/IIT/rtavarone/Data/PreProcessEcomodeItalian/DELTAS_TFRECORDS/TEST_FEAT_IREC_NUMPY/label_*.npy'))


test_features=sorted(glob.glob('/home/local/IIT/rtavarone/Data/PreProcessEcomodeItalian/DELTAS_TFRECORDS/ANTONIO_FEAT_TEST/features_*.npy'))
test_labels=sorted(glob.glob('/home/local/IIT/rtavarone/Data/PreProcessEcomodeItalian/DELTAS_TFRECORDS/ANTONIO_FEAT_TEST/label_*.npy'))


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
	prediction=tf.argmax(output_layer, axis=2)

	correct = tf.equal(prediction, labels)

	accuracy = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")

	outfile=open('max_posteriors_testset.csv','w')
	outfile.write('label,prediction,max_posterior,ratio_1,ratio_2,accuracy\n')

	for feat_file,label_file in zip(test_features,test_labels):
		data=np.load(feat_file,allow_pickle=False)
		data_length = np.array([data.shape[0]])
		data=np.expand_dims(data,axis=0)

		test_labels= np.load(label_file,allow_pickle=False)
		test_labels = np.expand_dims(test_labels,axis=0)
		leb, pred , post, acc = sess.run([labels,prediction,posteriors, accuracy], 
						feed_dict={input_layer: data, sequence_length: data_length, labels: test_labels})

		basename_feat=os.path.basename(feat_file)
		basename_label=os.path.basename(label_file)

		print('feature from {}   ----   label from {}'.format(basename_feat,basename_label))
		print('posteriors: ',post)
		print('label: {} , prediction: {} , accuracy: {}'.format(leb,pred,acc))
		print('')
		print('')
		overall_accuracy+=acc

		post=post[0][0]
		sort_post=sorted(post)

		outstring='{},{},{},{},{},{}\n'.format(leb[0],pred[0][0],sort_post[-1],sort_post[-1]/sum(sort_post[-2:-6:-1]),sort_post[-1]/sum(sort_post[-1:-6:-1]),acc)
		outfile.write(outstring)

print('overall test accuracy = ',overall_accuracy/len(test_features))

