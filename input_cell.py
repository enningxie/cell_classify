import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

IMAGE_SIZE = 24
tf.app.flags.DEFINE_string('train_dir', '/home/ck/cell_DL/cell_classify/CELL_images/train_data_trans', 'The dir of train')
tf.app.flags.DEFINE_integer('batch_size', 32, "the number of batch_size")
SPECIES = ['HSIL', 'LSIL', 'NILM']

def process(train_dir, batch_size):
	# 显示所有的图片目录
	img = tf.gfile.ListDirectory(train_dir)
	for im_ in img:
		if im_.split('_')[0] == 'HSIL':
			label = 0
		if im_.split('_')[0] == 'LSIL':
			label = 1
		if im_.split('_')[0] == 'NILM':
			label = 2
		img_dir = os.path.join(train_dir, im_)
		img_tensor = tf.image.decode_image(contents=img_dir, channels=3)
		img_float_tensor = tf.cast(img_tensor, tf.float32)

		height = IMAGE_SIZE
		width = IMAGE_SIZE
		resize_img = tf.image.resize_image_with_crop_or_pad(img_float_tensor, height, width)
		# 标准化图片
		float_image = tf.image.per_image_standardization(resize_img)
		float_image.set_shape([height, width, 3])
		labels = tf.cast(label, tf.int32)
		return _generate_images_label_batch(float_image, labels, batch_size, shuffle=True)


def _generate_images_label_batch(image, labels, batch_size, shuffle=True):
	if shuffle:
		images_batch, label_batch = tf.train.shuffle_batch(
			[image, labels],
			batch_size=batch_size,
			num_threads=8,
			capacity=5000,
			min_after_dequeue=1000
		)
	return images_batch, tf.reshape(label_batch, [batch_size])

def main(argv=None):
	if tf.gfile.Exists(FLAGS.train_dir):
		process(FLAGS.train_dir, FLAGS.batch_size)


if __name__ == '__main__':
	tf.app.run()