# Use a trained Image Transform Net to generate
# a style transferred image with a specific style

import tensorflow as tf
import image_transform_net as itn

from utils import get_image, get_batch_images


def generate(contents_path, model_path):
	if isinstance(contents_path, str):
		content_target = get_image(contents_path)
	elif len(contents_path) == 1:
		content_target = get_image(contents_path[0])
	else:
		content_target = get_batch_images(contents_path)

	content_images = tf.placeholder(tf.float32, shape=content_target.shape, name='content_images')

	output_images = itn.transform(content_images / 255.0)

	saver = tf.train.Saver()

	with tf.Session() as sess:
		saver.restore(sess, model_path)

		output = sess.run(output_images, feed_dict={content_images: content_target})

		return output

