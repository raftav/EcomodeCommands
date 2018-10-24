# Avoid printing tensorflow log messages
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import itertools
import numpy as np
import time
import sys

import lstm_model 


#################################
# Useful constant and paths
#################################
ExpNum = 1
restore_epoch = 34

class Configuration(object):
	
	audio_feat_dimension = 72

	audio_labels_dim=20

	n_hidden=100
	num_layers=3

checkpoints_dir='/home/local/IIT/rtavarone/Ecomode/EcomodeItalianDeltasOnlyIREC/BiLSTM_dropout/checkpoints/exp'+str(ExpNum)+'/'


def save_graph():

	config=Configuration()

	#priors = np.load('/DATA_NEW/rtavarone/Ecomode_Sentences_2018/unbalanced_noINEC/TrainingSetPriors.npy',allow_pickle=False)

	with tf.Graph().as_default():

		with tf.device('/cpu:0'):
			with tf.name_scope('inputs'):
				features = tf.placeholder(dtype=tf.float32, shape=[1,None,config.audio_feat_dimension],name='I')
				labels = tf.placeholder(dtype=tf.float32,shape=[1,1])
				sequence_length = tf.placeholder(dtype=tf.int32,shape=[1],name='LEN')

		with tf.device('/cpu:0'):
			with tf.variable_scope('model'):
				model = lstm_model.Model(features,labels,sequence_length,config,is_training=False)

		init_op = tf.local_variables_initializer()

		saver = tf.train.Saver(max_to_keep=None)

		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
											  log_device_placement=False)) as sess:

			sess.run(init_op)
			print('Restoring variables...')
			saver.restore(sess,checkpoints_dir+'model_epoch'+str(restore_epoch)+'.ckpt')
			print('Model loaded')

			save_path = saver.save(sess,'/home/local/IIT/rtavarone/Ecomode/EcomodeItalianDeltasOnlyIREC/BiLSTM_dropout_export/model_exp'+str(ExpNum)+'_epoch'+str(restore_epoch)+'.ckpt')
			tf.train.write_graph(sess.graph_def,'/home/local/IIT/rtavarone/Ecomode/EcomodeItalianDeltasOnlyIREC/BiLSTM_dropout_export/', 'model_exp'+str(ExpNum)+'_epoch'+str(restore_epoch)+ '.pbtxt')


def main(argv=None):  # pylint: disable=unused-argument
  save_graph()

if __name__ == '__main__':
  tf.app.run()

