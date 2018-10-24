from __future__ import print_function
from __future__ import division

# Avoid printing tensorflow log messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import time
import sys
import math
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import skip_gru_model 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for tick in plt.gca().xaxis.get_majorticklabels():
    	tick.set_horizontalalignment("right")


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    '''
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
	'''
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('ConfusionMatrix_exp{}_epoch{}'.format(ExpNum,restore_epoch))



#################################
# Useful constant and paths
#################################
ExpNum = int(sys.argv[1])
restore_epoch=int(sys.argv[2])

print('Evaluation at epoch ',restore_epoch)

class Configuration(object):

	logfile= open('TrainingExperiment'+str(ExpNum)+'.txt','r')
	lines = logfile.readlines()
	paramlines = [line for line in lines if line.startswith('##')]

	for params in paramlines:

		if 'learning rate' in params and 'update' not in params and 'decay' not in params:
			learning_rate = float (params.split(':',1)[1].strip() )
			print('learning_rate=',learning_rate)
		

		if 'optimizer' in params:
			optimizer_choice = params.split(':',1)[1].strip()
			print('optimizer =',optimizer_choice)

		if 'batch size' in params:
			batch_size=int(params.split(':',1)[1].strip())
			print('batch size =',batch_size)

		if 'number of hidden layers' in params:
			num_layers=int(params.split(':',1)[1].strip())
			print('n hidden layers=',num_layers)

		if 'number of hidden units' in params:
			n_hidden=int(params.split(':',1)[1].strip())
			print('n hidden =',n_hidden)

	slope_annealing_rate=0.005
	updating_step=184
	learning_decay=1.0
	keep_prob=1.0

	audio_feat_dimension = 144
	num_classes = 20
	num_examples_val=165


checkpoints_dir='checkpoints/exp'+str(ExpNum)+'/'


def read_my_file_format(filename_queue,feat_dimension=123):

    reader = tf.TFRecordReader()

    key, serialized_example = reader.read(filename_queue)

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example,
                                               context_features={"length": tf.FixedLenFeature([],dtype=tf.int64)},
                                               sequence_features={"audio_visual_feat":tf.FixedLenSequenceFeature([feat_dimension],dtype=tf.float32),
                                                                  "audio_labels":tf.FixedLenSequenceFeature([],dtype=tf.float32)}
                                        )

    return context_parsed['length'],sequence_parsed['audio_visual_feat'],tf.to_int32(sequence_parsed['audio_labels'])

def input_pipeline_validation(filenames,config):

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=1, shuffle=False)
    
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

def eval():

	config=Configuration()

	# list of input filenames + check existence
	filename_val=['/home/local/IIT/rtavarone/Data/PreProcessEcomode/MultiModalData/HIGHDIM_FEAT/TEST/sequence_full_{:04d}.tfrecords'.format(i) \
					 for i in range(config.num_examples_val)]
	for f in filename_val:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)
			

	with tf.Graph().as_default():

		# extract batch examples
		with tf.device('/cpu:0'):
			with tf.name_scope('validation_batch'):
				audio_features_val, audio_labels_val, seq_length_val = input_pipeline_validation(filename_val,config)

		with tf.device('/cpu:0'):
			with tf.variable_scope('model'):
				print('Building validation model:')
				val_model = skip_gru_model.Model(audio_features_val,audio_labels_val,seq_length_val,config,is_training=None)
				print('done.')

		# variables initializer
		print('Initializer creation')
		start=time.time()
		init_op = tf.local_variables_initializer()
		print('creation time = ',time.time() - start)
		print('Done')
		# save and restore all the variables.
		saver = tf.train.Saver(max_to_keep=None)

		# start session
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
											  log_device_placement=False)) as sess:

			# tensorboard writer
			#train_writer = tf.summary.FileWriter(tensorboard_dir,sess.graph)

			# run initializer
			
			sess.run(init_op)
			print('Restoring variables...')
			saver.restore(sess,checkpoints_dir+'model_epoch'+str(restore_epoch)+'.ckpt')
			print('variables loaded from ',tf.train.latest_checkpoint(checkpoints_dir))

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess = sess, coord = coord)

			print('')
			print('## EXPERIMENT NUMBER ',ExpNum)
			print('## binary units : deterministic')	
			print('## optimizer : ',config.optimizer_choice)
			print('## number of hidden layers : ',config.num_layers)
			print('## number of hidden units : ',config.n_hidden)
			print('## learning rate : ',config.learning_rate)
			print('## learning rate update steps: ',config.updating_step)
			print('## learning rate decay : ',config.learning_decay)
			print('## slope_annealing_rate : ',config.slope_annealing_rate)	
			print('## dropout keep prob (no dropout if 1.0) :',config.keep_prob)	
			print('## batch size : ',config.batch_size)		
			print('')

			try:
				
				accuracy=0.0
				layer_average= 0.0
				sentence=0
				labels=[]
				predictions=[]

				while not coord.should_stop():

					print('Evaluation on sentence {}'.format(sentence))

					example_accuracy , val_label , val_prediction,\
					gates , activations = sess.run([val_model.accuracy,
												val_model.labels,
												val_model.prediction,
												val_model.binary_states_fw,
												val_model.activations_norm_fw])

					labels.append(val_label[0,0])
					predictions.append(val_prediction[0,0])

					accuracy += example_accuracy
					gates = gates['z_5']
					activations = activations['5']

					sentece_length=gates.shape[1]
					layer_average += ( np.sum(gates) / sentece_length)

					print('Sentence length       = {}'.format(sentece_length))
					print('Number of updates     = {}'.format(np.sum(gates)))
					print('Percentage of updates = {}'.format(( np.sum(gates) / sentece_length)))
					print('label[{}]	prediction[{}]	accuracy[{}]'.format(val_label,val_prediction,example_accuracy))
					for frame in range(sentece_length):
						print('	gate[{}] activation[{}]'.format(gates[0,frame,0],activations[0,frame]))

					print('')

					sentence+=1

			except tf.errors.OutOfRangeError:
				accuracy /= config.num_examples_val
				layer_average /= config.num_examples_val

				print('Overall accuracy = ',accuracy)
				print('Overall percentage of updated = ',layer_average)

				labels = np.asarray(labels)
				predictions = np.asarray(predictions)

				cnf_matrix = confusion_matrix(labels, predictions)
				classes = []
				for i in range(19):
					classes.append( 'IPEC_{:03d}'.format(i+1) )
				classes.append('IREC_or_INEC')

				plot=plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                      title='Normalized confusion matrix')


def main(argv=None):  # pylint: disable=unused-argument
  eval()

if __name__ == '__main__':
  tf.app.run()