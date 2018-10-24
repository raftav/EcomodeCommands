from __future__ import division
import tensorflow as tf
import rnn_ops
import skip_rnn_cells
import layer_norm as layers


from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops

def variable_summaries(var,var_name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(var_name+'_summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


class Model(object):

	# layer normalization
	def _norm(self,inp, scope , norm_gain=1.0, norm_shift=0.0):
		shape = inp.get_shape()[-1:]
		gamma_init = init_ops.constant_initializer(norm_gain)
		beta_init = init_ops.constant_initializer(norm_shift)

		with vs.variable_scope(scope):
			# Initialize beta and gamma for use by layer_norm.
			vs.get_variable("gamma", shape=shape, initializer=gamma_init)
			vs.get_variable("beta", shape=shape, initializer=beta_init)

		normalized = layers.layer_norm(inp, reuse=True, scope=scope)
		return normalized

	def __init__(self,features,labels,seq_length,config,is_training):

		# batch size cannot be inferred from features shape because
		# it must be defined statically
		if is_training:
			batch_size=config.batch_size
		else:
			batch_size=1

		# global step for learning rate decay
		global_step = tf.Variable(0,name='global_step', trainable=False)
		self._global_step=global_step

		# slope of the sigmoid for slope annealing trick
		slope = tf.to_float(global_step / config.updating_step) * tf.constant(config.slope_annealing_rate) + tf.constant(1.0)
		self._slope = slope

		# stack of custom rnn cells
		num_units = [config.n_hidden for _ in range(config.num_layers)]
		with tf.variable_scope('forward_cells'):
			multi_cell_fw = skip_rnn_cells.MultiSkipGRUCell(num_units,layer_norm=True)
			initial_state_fw = multi_cell_fw.trainable_initial_state(batch_size)
	

		# linear mapping of features dimension to dimension of
		# first hidden layer
		with tf.variable_scope('embedding'):
			embedding_weights = tf.get_variable('embedding_weights',
								[config.audio_feat_dimension,config.n_hidden],
								initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1))
			
			features = tf.reshape(features,[-1,config.audio_feat_dimension])

			embedded_input = tf.matmul(features,embedding_weights)

			embedded_input = self._norm(embedded_input,"input")
			embedded_input = tf.reshape(embedded_input,[batch_size,-1,config.n_hidden])
		


		# FORWARD RNN
		# use dynamic_rnn for training
		with tf.variable_scope('forward_rnn'):
			rnn_outputs, last_state_fw = tf.nn.dynamic_rnn(multi_cell_fw,embedded_input,
														sequence_length=seq_length,
														initial_state=initial_state_fw)

		rnn_outputs, updated_states = rnn_outputs.h, rnn_outputs.state_gate
				
		if not is_training:

				activation={}
				states={}

				i=config.num_layers

				states['z_{:d}'.format(i)] = updated_states
				activation['{}'.format(i)] = tf.norm(rnn_outputs,ord=1,axis=2)

		with tf.variable_scope("Output"):
			
			rnn_output = tf.slice(rnn_outputs,[0,tf.shape(rnn_outputs)[1]-1,0],[-1,-1,-1])

			output_weights = tf.get_variable('W_out',shape=[config.n_hidden,config.num_classes],
											initializer=tf.orthogonal_initializer())

			output_bias = tf.get_variable('b_out',shape=[config.num_classes],initializer=tf.ones_initializer())

			rnn_output = tf.reshape(rnn_output,[-1,config.n_hidden])

			output = tf.matmul(rnn_output,output_weights) + output_bias
				
			# shape back to [batch_size, max_time, num_classes]
			logits = tf.reshape(output,shape=[batch_size,-1,config.num_classes])
		
		prediction=tf.argmax(logits, axis=2)
		self._prediction = prediction
		self._labels = labels

		if is_training:

			# evaluate cost and optimize
			with tf.name_scope('cost'):

				all_states = tf.reduce_sum(updated_states)

				cross_entropy_loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels))

				l2_loss = config.lambda_l2 * all_states

				self._cost = cross_entropy_loss + l2_loss

				tf.summary.scalar('cost',self._cost)

			with tf.name_scope('optimizer'):
				learning_rate = tf.train.exponential_decay(config.learning_rate, global_step,config.updating_step, config.learning_decay, staircase=True)
				self._learning_rate= learning_rate

				if 'momentum' in config.optimizer_choice:
					self._optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)

				elif 'adam' in config.optimizer_choice:
					self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

				# gradient clipping
				
				gradients , variables = zip(*self._optimizer.compute_gradients(self._cost))
				clip_grad  = [None if gradient is None else \
							tf.clip_by_norm(gradient, 1.0) \
							for gradient in gradients]
							 
				self._optimize = self._optimizer.apply_gradients(zip(clip_grad,variables),global_step=self._global_step)
				

		else:

			correct = tf.equal(prediction,tf.to_int64(labels))
			self._accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))


			self._binary_states_fw = states
			self._activations_norm_fw = activation



	@property
	def cost(self):
		return self._cost

	@property
	def optimize(self):
		return self._optimize

	@property
	def cell_slope(self):
		return self._cell_slope

	@property
	def prediction(self):
		return self._prediction

	@property
	def output_logits(self):
		return self._output_logits
		x
	@property
	def accuracy(self):
		return self._accuracy

	@property
	def binary_states_fw(self):
		return self._binary_states_fw

	@property
	def binary_logits_fw(self):
		return self._binary_logits_fw

	@property
	def activations_norm_fw(self):
		return self._activations_norm_fw

		
	@property
	def binary_states_bw(self):
		return self._binary_states_bw

	@property
	def binary_logits_bw(self):
		return self._binary_logits_bw

	@property
	def activations_norm_bw(self):
		return self._activations_norm_bw

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
	def slope(self):
		return self._slope

	@property
	def decoded(self):
		return self._decoded