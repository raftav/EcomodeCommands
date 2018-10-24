from __future__ import division
import tensorflow as tf
import sys

class Model(object):

	def __init__(self,features,labels,seq_length,config,is_training):

		if is_training:
			batch_size=config.batch_size
		else:
			batch_size=1

		global_step = tf.Variable(0, trainable=False)
		self._global_step=global_step

		# lstm cells definition
		with tf.variable_scope('forward'):

			forward_cells = []
			forward_cells_init = []
			for i in range(config.num_layers):
				with tf.variable_scope('layer_{:d}'.format(i)):
					lstm_cell_forward = tf.contrib.rnn.LSTMCell(config.n_hidden,use_peepholes=False,
											forget_bias=1.0,activation=tf.tanh,
											initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
					forward_cells.append(lstm_cell_forward)
					forward_cells_init.append(lstm_cell_forward.zero_state(batch_size,dtype=tf.float32))

		with tf.variable_scope('backward'):

			backward_cells = []
			backward_cells_init = []
			for i in range(config.num_layers):
				with tf.variable_scope('layer_{:d}'.format(i)):
					lstm_cell_backward = tf.contrib.rnn.LSTMCell(config.n_hidden,use_peepholes=False,
											forget_bias=1.0,activation=tf.tanh,
											initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
					backward_cells.append(lstm_cell_backward)
					backward_cells_init.append(lstm_cell_backward.zero_state(batch_size,dtype=tf.float32))

		with tf.variable_scope('RNN'):
			rnn_outputs, output_state_fw , output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
											cells_fw=forward_cells,
											cells_bw=backward_cells,
											inputs=features,
											initial_states_fw=forward_cells_init,
											initial_states_bw=backward_cells_init,
											dtype=tf.float32,
											sequence_length=seq_length)

			rnn_last_output_fw=output_state_fw[-1].h
			rnn_last_output_bw=tf.slice(rnn_outputs,[0,0,config.n_hidden],[-1,1,-1])
			rnn_last_output_bw=tf.squeeze(rnn_last_output_bw,axis=1)

			rnn_output = tf.concat((rnn_last_output_fw,rnn_last_output_bw),axis=-1)

		with tf.variable_scope('output'):
			output_fw_weights = tf.get_variable('forward_weights',[2*config.n_hidden,config.audio_labels_dim],dtype=tf.float32,
									initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
			output_biases = tf.get_variable('biases',shape=[config.audio_labels_dim],dtype=tf.float32,
											initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))

			rnn_output = tf.reshape(rnn_output,[-1,2*config.n_hidden])	
			
			output = tf.matmul(rnn_output,output_fw_weights) + output_biases
		
			logits = tf.reshape(output,[batch_size,-1,config.audio_labels_dim])

		if is_training:
		# evaluate cost and optimize
			with tf.name_scope('cost'):
				self._cost = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels))
				tf.summary.scalar('cost',self._cost)

			with tf.name_scope('optimizer'):
				#learning_rate = tf.train.exponential_decay(config.learning_rate, global_step,
				#                    config.updating_step, config.learning_decay, staircase=True)

				learning_rate = config.learning_rate
				self._learning_rate= learning_rate

				if "momentum" in config.optimizer_choice:
					self._optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
				elif "adam" in config.optimizer_choice:
					self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
				else:
					print("Optimizer must be either momentum or adam. Closing.")
					sys.exit()

				# gradient clipping
				gradients , variables = zip(*self._optimizer.compute_gradients(self._cost))
				clip_grad  = [tf.clip_by_norm(gradient, 1.0) for gradient in gradients] 
				self._optimize = self._optimizer.apply_gradients(zip(clip_grad,variables),global_step=self._global_step)


		posteriors=tf.nn.softmax(logits)
		prediction=tf.argmax(logits, axis=2)
		correct = tf.equal(prediction,tf.to_int64(labels))
		accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

		self._posteriors=posteriors
		self._accuracy=accuracy
		self._labels = labels
		self._prediction = prediction

	@property
	def out_before_slice(self):
		return self._out_before_slice


	@property
	def out_after_slice(self):
		return self._out_after_slice
		
	@property
	def cost(self):
		return self._cost

	@property
	def optimize(self):
		return self._optimize

	@property
	def posteriors(self):
		return self._posteriors

	@property
	def correct(self):
		return self._correct

	@property
	def accuracy(self):
		return self._accuracy

	@property
	def labels(self):
		return self._labels

	@property
	def learning_rate(self):
		return self._learning_rate

	@property
	def global_step(self):
		return self._global_step


	@property
	def prediction(self):
		return self._prediction