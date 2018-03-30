import tensorflow as tf

scope_name = 'lol'
with tf.name_scope(name=scope_name) as scope:
    a = tf.get_variable(name=scope, shape=[5, 5], initializer=tf.constant_initializer(0.0))
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(a.op.name)
    print(a.get_shape().as_list())