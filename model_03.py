# 03_03 raw model for cnn
import tensorflow as tf
import utils
import argparse
import time


# parse_params
def parse_args():
    parser = argparse.ArgumentParser(description='something for config the model.')
    parser.add_argument('--lr', help='learning rate', default=0.001, type=float)
    return parser.parse_args()


# raw cnn_op
def conv_op(inputs, filters, ksize, stride, scope_name, activation=True, padding='SAME'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_channel = inputs.shape[-1]
        kernel = tf.get_variable('w', shape=[ksize, ksize, in_channel, filters], initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('b', shape=[filters], initializer=tf.constant_initializer(0.0))
        # Must have `strides[0] = strides[3] = 1`.
        conv = tf.nn.conv2d(inputs, filter=kernel, strides=[1, stride, stride, 1], padding=padding)
        outputs = tf.nn.bias_add(conv, biases)
        if activation:
            outputs = tf.nn.relu(outputs)
    return outputs


# raw max_pool_op
def max_pool_op(inputs, ksize, stride, scope_name, padding='VALID'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(inputs, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)
    return pool


# full connect_op
def connect_op(inputs,  dim_out, scope_name, activation=True):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        # inputs_shape = inputs.shape
        # reshape_in = tf.reshape(inputs, [-1, inputs_shape[1]*inputs_shape[2]*inputs_shape[3]])
        dim_in = inputs.shape[-1]
        w = tf.get_variable('w_1', shape=[dim_in, dim_out], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('b_1', shape=[dim_out], initializer=tf.random_normal_initializer())
        outputs = tf.matmul(inputs, w) + b
        if activation:
            outputs = tf.nn.relu(outputs)
    return outputs


class Model(object):
    def __init__(self, args):
        self.lr = args.lr
        self.gstep = tf.Variable(0, dtype=tf.int32,
                                 trainable=False, name='global_step')
        self.training = True
        self.skip_step = 5
        self.n_test = 545
        self.batch_size = 32
        self.n_classes = 3

    def get_data(self):
        with tf.name_scope('data'):
            train_data, test_data = utils.process_cell_data(self.batch_size)
            iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                       train_data.output_shapes)
            img, self.label = iterator.get_next()
            self.img = tf.reshape(img, shape=[-1, 32, 32, 3])
            # reshape the image to make it work with tf.nn.conv2d

            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)  # initializer for train_data

    def print_info(self, t):
        print(t.op.name, ' ', t.get_shape().as_list())

    def inference(self):
        # ---------------------------------------------------
        # conv layer 1
        conv1 = conv_op(inputs=self.img,
                           filters=32,
                           ksize=3,
                           stride=1,
                           scope_name='conv1')

        # maxpool1
        pool1 = max_pool_op(inputs=conv1,
                        ksize=3,
                        stride=1,
                        scope_name='pool1')

        # show info
        self.print_info(pool1)

        # ---------------------------------------------------

        # conv layer 2
        conv2 = conv_op(inputs=pool1,
                           filters=64,
                           ksize=5,
                           stride=1,
                           scope_name='conv2')

        # maxpool2
        pool2 = max_pool_op(inputs=conv2,
                        ksize=5,
                        stride=1,
                        scope_name='pool2')

        # show
        self.print_info(pool2)

        # ---------------------------------------------------

        # conv layer 3
        conv3 = conv_op(inputs=pool2,
                           filters=64,
                           ksize=3,
                           stride=1,
                           scope_name='conv3')

        # show
        self.print_info(conv3)

        # ---------------------------------------------------

        # conv layer 4
        conv4 = conv_op(inputs=conv3,
                           filters=64,
                           ksize=3,
                           stride=1,
                           scope_name='conv4')

        # show
        self.print_info(conv4)

        # ---------------------------------------------------

        # conv layer 5
        conv5 = conv_op(inputs=conv4,
                           filters=64,
                           ksize=3,
                           stride=1,
                           scope_name='conv5')

        # maxpool
        pool5 = max_pool_op(inputs=conv5,
                        ksize=3,
                        stride=1,
                        scope_name='pool5')

        # show
        self.print_info(pool5)

        # ---------------------------------------------------
        tmp_shape = pool5.get_shape().as_list()
        fc1_input = tf.reshape(pool5, [-1, tmp_shape[1] * tmp_shape[2] * tmp_shape[3]])
        fc1 = connect_op(inputs=fc1_input,
                         dim_out=128,
                              scope_name='fc1')
        # ---------------------------------------------------
        fc2 = connect_op(inputs=fc1,
                         dim_out=64,
                              scope_name='fc2')
        # ---------------------------------------------------
        self.logits = connect_op(inputs=fc2,
                                 dim_out=self.n_classes,
                                      activation=False,
                                      scope_name='logits')

    def loss_op(self):
        with tf.name_scope('loss'):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label)
            self.loss = tf.reduce_mean(entropy, name='loss_op')

    def optimize(self):
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)

    def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), self.label)
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build(self):
        '''
        Build the computation graph
        '''
        self.get_data()
        self.inference()
        self.loss_op()
        self.optimize()
        self.eval()

    def train_one_epoch(self, sess, init, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l= sess.run([self.opt, self.loss])

                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch= sess.run(self.accuracy)

                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / self.n_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, self.train_init, epoch, step)
                self.eval_once(sess, self.test_init, epoch, step)


if __name__ == '__main__':
    args = parse_args()
    model = Model(args)
    model.build()
    model.train(n_epochs=5)
