from __future__ import print_function
from __future__ import division

# Avoid printing tensorflow log messages
import os
import tensorflow as tf
from tensorflow.python import debug as tf_debug

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import numpy as np
import time
import sys

import lstm_model 

#################################
# Useful constant and paths
#################################
ExpNum = int(sys.argv[1])

num_examples=1432

class Configuration(object):
	
	learning_rate=float(sys.argv[2])
	batch_size=int(sys.argv[3])
	optimizer_choice=sys.argv[4]

	audio_feat_dimension = 24

	audio_labels_dim=20
	
	num_epochs=5000
	
	n_hidden=100
	num_layers=5

	num_examples_val=338

checkpoints_dir='checkpoints/exp'+str(ExpNum)+'/'

tensorboard_dir='tensorboard/exp'+str(ExpNum)+'/'

trainingLogFile=open('TrainingExperiment'+str(ExpNum)+'.txt','w')

if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)


###################################
# Auxiliary functions
###################################

# Reads a single serialized SequenceExample
def read_my_file_format(filename_queue,feat_dimension=123):

    reader = tf.TFRecordReader()

    key, serialized_example = reader.read(filename_queue)

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example,
                                               context_features={"length": tf.FixedLenFeature([],dtype=tf.int64)},
                                               sequence_features={"audio_feat":tf.FixedLenSequenceFeature([feat_dimension],dtype=tf.float32),
                                                                  "audio_labels":tf.FixedLenSequenceFeature([],dtype=tf.float32)}
                                        )

    return context_parsed['length'],sequence_parsed['audio_feat'],tf.to_int32(sequence_parsed['audio_labels'])

# training input pipeline
def input_pipeline(filenames,config):

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=config.num_epochs, shuffle=True)
    
    sequence_length, audio_features, audio_labels = read_my_file_format(filename_queue,feat_dimension=config.audio_feat_dimension)

    audio_features_batch, audio_labels_batch , seq_length_batch = tf.train.batch([audio_features, audio_labels, sequence_length],
                                                    batch_size=config.batch_size,
                                                    num_threads=10,
                                                    capacity=100,
                                                    dynamic_pad=True,
                                                    enqueue_many=False)

    return audio_features_batch, audio_labels_batch, seq_length_batch

def input_pipeline_validation(filenames,config):

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None, shuffle=False)
    
    sequence_length, audio_features, audio_labels = read_my_file_format(filename_queue,feat_dimension=config.audio_feat_dimension)

    audio_features_batch, audio_labels_batch , seq_length_batch = tf.train.batch([audio_features, audio_labels, sequence_length],
                                                    batch_size=1,
                                                    num_threads=10,
                                                    capacity=100,
                                                    dynamic_pad=True,
                                                    enqueue_many=False)

    return audio_features_batch, audio_labels_batch, seq_length_batch

#################################
# Training module
#################################
def train():

	config=Configuration()

	# list of input filenames + check existence
	filename_train=['/home/local/IIT/rtavarone/Data/PreProcessEcomode/RecurrentData_NoDeltas/TRAIN/sequence_full_{:04d}.tfrecords'.format(i) \
					 for i in range(num_examples)]
	for f in filename_train:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	filename_val=['/home/local/IIT/rtavarone/Data/PreProcessEcomode/RecurrentData_NoDeltas/VAL/sequence_full_{:04d}.tfrecords'.format(i) \
					 for i in range(config.num_examples_val)]
	for f in filename_train:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)


	# training graph		
	with tf.Graph().as_default():

		# extract batch examples
		with tf.device('/cpu:0'):
			with tf.name_scope('train_batch'):
				audio_features, audio_labels, seq_length = input_pipeline(filename_train,config)

		
		with tf.device('/cpu:0'):
			with tf.name_scope('validation_batch'):
				audio_features_val, audio_labels_val, seq_length_val = input_pipeline_validation(filename_val,config)
		
		# audio features reconstruction
		with tf.device('/cpu:0'):
			with tf.variable_scope('model',reuse=None):
				print('Building training model:')
				train_model = lstm_model.Model(audio_features,audio_labels,seq_length,config,is_training=True)
				print('done.\n')

		
		with tf.device('/cpu:0'):
			with tf.variable_scope('model',reuse=True):
				print('Building validation model:')
				val_model = lstm_model.Model(audio_features_val,audio_labels_val,seq_length_val,config,is_training=None)
				print('done.')
		

		# variables initializer
		init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

		# save and restore all the variables.
		saver = tf.train.Saver(max_to_keep=10)

		
		# start session
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
											  log_device_placement=False)) as sess:

			# tensorboard writer
			#train_writer = tf.summary.FileWriter(tensorboard_dir,sess.graph)

			# run initializer
			sess.run(init_op)


			# start queue coordinator and runners
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess = sess, coord = coord)


			#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
			#sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

			print('')
			print('## EXPERIMENT NUMBER ',ExpNum)
			trainingLogFile.write('# EXPERIMENT NUMBER {:d} \n'.format(ExpNum))
			
			print('## optimizer : ',config.optimizer_choice)
			trainingLogFile.write('## optimizer : {:s} \n'.format(config.optimizer_choice))
			
			print('## number of hidden layers : ',config.num_layers)
			trainingLogFile.write('## number of hidden layers : {:d} \n'.format(config.num_layers))
			
			print('## number of hidden units : ',config.n_hidden)
			trainingLogFile.write('## number of hidden units : {:d} \n'.format(config.n_hidden))
			
			print('## learning rate : ',config.learning_rate)
			trainingLogFile.write('## learning rate : {:.6f} \n'.format(config.learning_rate))
			
			print('## batch size : ',config.batch_size)
			trainingLogFile.write('## batch size : {:d} \n'.format(config.batch_size))
			
			print('## number of steps: ',num_examples*config.num_epochs/config.batch_size)
			trainingLogFile.write('## approx number of steps: {:d} \n'.format(int(num_examples*config.num_epochs/config.batch_size)))
			
			print('## number of steps per epoch: ',num_examples/config.batch_size)
			trainingLogFile.write('## approx number of steps per epoch: {:d} \n'.format(int(num_examples/config.batch_size)))
			print('')

			try:
				epoch_counter=1
				epoch_cost=0.0
				EpochStartTime=time.time()

				step=1

				while not coord.should_stop():

					_ , C  , train_pred , train_label ,\
					out_before_slice, out_after_slice  = sess.run([train_model.optimize,train_model.cost,
														train_model.prediction,
														train_model.labels,
														train_model.out_before_slice,
														train_model.out_after_slice])


					epoch_cost += C

					if (step % 50 == 0 or step==1):
						print("step[{:7d}] cost[{:2.5f}] ".format(step,C))
						'''
						print('rnn out before slice shape: ',out_before_slice.shape)
						print('last rnn out before slice: ',out_before_slice[0,-1,:])

						print('rnn out after slice shape: ',out_after_slice.shape)
						print('rnn out after slice: ',out_after_slice)
						'''
					'''
					if (step%500==0):
						# save training parameters
						save_path = saver.save(sess,checkpoints_dir+'model_step'+str(step)+'.ckpt')
						print('Model saved!')
					'''

					if ((step % int(num_examples / config.batch_size) == 0) and (step is not 0)):

						# at each step we get the average cost over a batch, so divide by the 
						# number of batches in one epoch
						epoch_cost /=  (num_examples/config.batch_size)

						print('Completed epoch {:d} at step {:d} --> cost[{:.6f}]'.format(epoch_counter,step,epoch_cost))
						print('Epoch training time (seconds) = ',time.time()-EpochStartTime)
						
						#accuracy evaluation on each sentence
						#to avoid computing accuracy on padded frames
						
						out_every_epoch=1
							
						if((epoch_counter%out_every_epoch)==0):

							accuracy=0.0

							for i in range(config.num_examples_val):

								# validation
								example_accuracy , val_label , val_prediction = sess.run([val_model.accuracy,
																			val_model.labels,
																			val_model.prediction])

								print('index[{}] label[{}] prediction[{}] accuracy[{}]'.format(i,val_label,val_prediction,example_accuracy))
								accuracy+=example_accuracy


							accuracy /= config.num_examples_val
							
							# printout validation results
							print('Validation accuracy : {} '.format(accuracy))
							
							trainingLogFile.write('{:d}\t{:.8f}\t{:.8f}\n'.format(epoch_counter,epoch_cost,accuracy))
							trainingLogFile.flush()

							save_path = saver.save(sess,checkpoints_dir+'model_epoch'+str(epoch_counter)+'.ckpt')
						
						print('\n')	
						epoch_counter+=1
						epoch_cost=0.0
						EpochStartTime=time.time()

					step += 1

			except tf.errors.OutOfRangeError:
				print('---- Done Training: epoch limit reached ----')
			finally:
				coord.request_stop()

			coord.join(threads)

			save_path = saver.save(sess,checkpoints_dir+'model_end.ckpt')
			print("model saved in file: %s" % save_path)

	trainingLogFile.close()


def main(argv=None):  # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()