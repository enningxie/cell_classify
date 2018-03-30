import tensorflow as tf
import time
import os
import input_cell


# conv layer
def conv_layer(inputs, filters, k_size, stride, scope_name, padding='SAME'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_channels = inputs.shape[-1]
        kernel = tf.get_variable('kernel', [k_size, k_size, in_channels, filters], initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [filters], initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)
    return tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)


# pool layer
def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(inputs, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)
    return pool


# fully connect layer
def fully_connected(inputs, out_dim, activation=True, scope_name='fc'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        w = tf.get_variable('weights', [in_dim, out_dim], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(tf.matmul(inputs, w), b)
        if activation:
            out = tf.nn.relu(out)
    return out


# oop
class Vgg16(object):
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 128
        self.keep_prob = tf.constant(0.75)
        self.gstep = tf.get_variable(name='global_step', shape=0, dtype=tf.int32, trainable=False)
        self.n_classes = 3
        self.skip_step = 20  # 用于展示间隔用
        self.model_path = ''  # 用于存储model
        self.n_test = 1000  # 用于test
        self.training = True
        self.graph_path = ''

    def print_info(self, t):
        print(t.op.name, " ", t.get_shape().as_list)

    def get_data(self):
        with tf.name_scope('data'):
            self.img = input_cell.inputs()[0]
            self.label = input_cell.inputs()[1]

    def inference(self):
        # ---------------------------------------------------
        # conv layer 1
        conv1 = conv_layer(inputs=self.img,
                           filters=64,
                           k_size=11,
                           stride=4,
                           scope_name='conv1')

        # maxpool1
        pool1 = maxpool(inputs=conv1,
                        ksize=3,
                        stride=2,
                        scope_name='pool1')

        # show info
        self.print_info(pool1)

        # ---------------------------------------------------

        # conv layer 2
        conv2 = conv_layer(inputs=pool1,
                           filters=192,
                           k_size=5,
                           stride=1,
                           scope_name='conv2')

        # maxpool2
        pool2 = maxpool(inputs=conv2,
                        ksize=3,
                        stride=2,
                        scope_name='pool2')

        # show
        self.print_info(pool2)

        # ---------------------------------------------------

        # conv layer 3
        conv3 = conv_layer(inputs=pool2,
                           filters=348,
                           k_size=3,
                           stride=1,
                           scope_name='conv3')

        # show
        self.print_info(conv3)

        # ---------------------------------------------------

        # conv layer 4
        conv4 = conv_layer(inputs=conv3,
                           filters=256,
                           k_size=3,
                           stride=1,
                           scope_name='conv4')

        # show
        self.print_info(conv4)

        # ---------------------------------------------------

        # conv layer 5
        conv5 = conv_layer(inputs=conv4,
                           filters=256,
                           k_size=3,
                           stride=1,
                           scope_name='conv5')

        # maxpool
        pool5 = maxpool(inputs=conv5,
                        ksize=3,
                        stride=2,
                        scope_name='pool5')

        # show
        self.print_info(pool5)

        # ---------------------------------------------------
        tmp_shape = pool5.get_shape().as_list()

        fc1_input = tf.reshape(pool5, [-1, tmp_shape[1]*tmp_shape[2]*tmp_shape[3]])
        fc1 = fully_connected(inputs=fc1_input,
                              out_dim=4096,
                              scope_name='fc1')
        # ---------------------------------------------------
        fc2 = fully_connected(inputs=fc1,
                              out_dim=4096,
                              scope_name='fc2')
        # ---------------------------------------------------
        self.logits = fully_connected(inputs=fc2,
                                      out_dim=self.n_classes,
                                      activation=False,
                                      scope_name='logits')

    def loss(self):
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')

    # optimizer
    def optimize(self):
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)

    def summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def eval(self):
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    # build the graph
    def build(self):
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    def train_one_step(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, self.model_path, step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, sess.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass
        print('Accuracy at epoch {0}: {1}'.format(epoch, total_correct_preds/self.n_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        writer = tf.summary.FileWriter(self.graph_path, tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.model_path))
            if ckpt and ckpt.model_checkpoint_path:
                saver.save(sess, ckpt.model_checkpoint_path)
            step = self.gstep
            for epoch in range(n_epochs):
                step = self.train_one_step(sess, saver, _, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
            writer.close()


if __name__ == '__main__':
    model = Vgg16()
    model.build()
    model.train(n_epochs=30)









