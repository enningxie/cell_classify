import tensorflow as tf
import input_cell

tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_epochs', 10, 'Number of training epochs')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Batch size')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
tf.app.flags.DEFINE_float('dropout_rate', 0.5, 'Dropout rate')
tf.app.flags.DEFINE_string('train_dataset', 'train.tfrecords',
                           'Filename of training dataset')
tf.app.flags.DEFINE_string('eval_dataset', 'eval.tfrecords',
                           'Filename of evaluation dataset')
tf.app.flags.DEFINE_string('test_dataset', 'test.tfrecords',
                           'Filename of testing dataset')
tf.app.flags.DEFINE_string('model_dir', 'models/cifar10_cnn_model',
                           'Filename of testing dataset')


def my_model(features, labels, mode):
    inputs = features
	# CONV1
	with tf.name_scope('CONV1'):
		conv1 = tf.layers.conv2d(inputs, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu, name="conv1")
		conv1_bn = tf.layers.batch_normalization(conv1, training=mode == tf.estimator.ModeKeys.TRAIN, name="BN1")

	# CONV2
	with tf.name_scope('CONV2'):
		conv2 = tf.layers.conv2d(conv1_bn, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu, name="conv2")
		conv2_bn = tf.layers.batch_normalization(conv2, training=mode == tf.estimator.ModeKeys.TRAIN, name="BN2")

	# pool1
	with tf.name_scope("POOL1"):
		pool1 = tf.layers.max_pooling2d(conv2_bn, pool_size=[2, 2], padding='same', strides=2, name='pool1')

	# CONV3
	with tf.name_scope("CONV3"):
		conv3 = tf.layers.conv2d(pool1, filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu, name="conv3")
		conv3_bn = tf.layers.batch_normalization(conv3, training=mode == tf.estimator.ModeKeys.TRAIN, name="BN3")

	# CONV4
	with tf.name_scope("CONV4"):
		conv4 = tf.layers.conv2d(conv3_bn, filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu, name="conv4")
		conv4_bn = tf.layers.batch_normalization(conv4, training=mode == tf.estimator.ModeKeys.TRAIN, name="BN4")

	# POOl2
	with tf.name_scope("pool2"):
		pool2 = tf.layers.max_pooling2d(conv4_bn, pool_size=[2, 2], padding='same', strides=2, name="pool2")

	# CONV5
	with tf.name_scope("CONV5"):
		conv5 = tf.layers.conv2d(pool2, filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
								 name="conv5")
		conv5_bn = tf.layers.batch_normalization(conv5, training=mode == tf.estimator.ModeKeys.TRAIN, name="BN5")

	# CONV6
	with tf.name_scope("CONV6"):
		conv6 = tf.layers.conv2d(conv5_bn, filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
								 name="conv6")
		conv6_bn = tf.layers.batch_normalization(conv6, training=mode == tf.estimator.ModeKeys.TRAIN, name="BN6")


	# CONV7
	with tf.name_scope("CONV7"):
		conv7 = tf.layers.conv2d(conv6_bn, filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
								 name="conv7")
		conv7_bn = tf.layers.batch_normalization(conv7, training=mode == tf.estimator.ModeKeys.TRAIN, name="BN7")

	# POOL3
	with tf.name_scope("pool3"):
		pool3 = tf.layers.max_pooling2d(conv7_bn, pool_size=[2, 2], padding='same', strides=2, name='POOL3')

	# CONV8
	with tf.name_scope("CONV8"):
		conv8 = tf.layers.conv2d(conv7_bn, filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
								 name="conv8")
		conv8_bn = tf.layers.batch_normalization(conv8, training=mode == tf.estimator.ModeKeys.TRAIN, name="BN8")

	# CONV9
	with tf.name_scope("CONV9"):
		conv9 = tf.layers.conv2d(conv8_bn, filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
								 name="conv9")
		conv9_bn = tf.layers.batch_normalization(conv9, training=mode == tf.estimator.ModeKeys.TRAIN, name="BN9")

	# CONV10
	with tf.name_scope("CONV10"):
		conv10 = tf.layers.conv2d(conv9_bn, filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
								 name="conv10")
		conv10_bn = tf.layers.batch_normalization(conv10, training=mode == tf.estimator.ModeKeys.TRAIN, name="BN10")

	# POOL4
	with tf.name_scope("pool4"):
		pool4 = tf.layers.max_pooling2d(conv10_bn, pool_size=[2, 2], padding='same', strides=2, name='POOL4')

	# CONV11
	with tf.name_scope("CONV11"):
		conv11 = tf.layers.conv2d(pool4, filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
								 name="conv11")
		conv11_bn = tf.layers.batch_normalization(conv11, training=mode == tf.estimator.ModeKeys.TRAIN, name="BN11")

	# CONV12
	with tf.name_scope("CONV12"):
		conv12 = tf.layers.conv2d(conv10_bn, filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
								 name="conv12")
		conv12_bn = tf.layers.batch_normalization(conv12, training=mode == tf.estimator.ModeKeys.TRAIN, name="BN12")

	# CONV13
	with tf.name_scope("CONV13"):
		conv13 = tf.layers.conv2d(conv12_bn, filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
								 name="conv13")
		conv13_bn = tf.layers.batch_normalization(conv13, training=mode == tf.estimator.ModeKeys.TRAIN, name="BN13")

	# POOL5
	with tf.name_scope("pool5"):
		pool5 = tf.layers.max_pooling2d(conv13_bn, pool_size=[2, 2], padding='same', strides=2, name='POOL4')

	tmp_shape = pool5.get_shape().as_list()
	with tf.name_scope("dense"):
		dense = tf.reshape(pool5, [-1, tmp_shape[1]*tmp_shape[2]*tmp_shape[3]])
		dense1 = tf.layers.dense(dense, units=4096, activation=tf.nn.relu, name="dense1")
		dense1_drop = tf.layers.dropout(dense1)
		dense2 = tf.layers.dense(dense1_drop, units=4096, activation=tf.nn.relu, name="dense2")
		dense2_drop = tf.layers.dropout(dense2)
		logits = tf.layers.dense(dense2_drop, units=3, name="final")

	# predict
	predictions = {
		'classes': tf.argmax(logits, axis=1, name='classes'),
		'probabilities': tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	onehot_label = tf.one_hot(labels, depth=3)
	# 计算损失
	loss = tf.losses.softmax_cross_entropy(onehot_label, logits, scope="LOSS")

	# tommorrow
	accuracy, update_op = tf.metrics.accuracy(
		labels=labels, predictions=predictions['classes'], name='accuracy')
	batch_acc = tf.reduce_mean(tf.cast(
		tf.equal(tf.cast(labels, tf.int64), predictions['classes']), tf.float32))
	tf.summary.scalar('batch_acc', batch_acc)
	tf.summary.scalar('streaming_acc', update_op)

	if mode == tf.estimator.ModeKeys.TRAIN:
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
		with tf.control_dependencies(update_ops):
			train_op = optimizer.minimize(
				loss=loss, global_step=tf.train.get_global_step()
			)
		return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss)


def main(unused_argv):
	cell_classifier = tf.estimator.Estimator(
		model_fn=my_model
		#model_dir=
	)
	cell_classifier.train(input_fn=input_cell.inputs)
	tf.logging.info('Saving hyperparameters ...')


if __name__ == '__main__':
	tf.app.run()
