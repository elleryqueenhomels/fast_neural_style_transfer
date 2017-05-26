# Use a trained Image Transform Net to generate
# a style transferred image with a specific style

import tensorflow as tf
import image_transform_net as itn

from utils import get_images, save_images


def generate(contents_path, model_path, is_same_size=False, resize_height=None, resize_width=None, save=True, postfix='-stylized'):
	if isinstance(contents_path, str):
		contents_path = [contents_path]

	if is_same_size or (resize_height is not None and resize_width is not None):
		outputs = _handler(contents_path, model_path, resize_height=resize_height, resize_width=resize_width, save=save, postfix=postfix)
		return [outputs[i] for i in range(len(outputs))]
	else:
		import numpy as np
		outputs = []
		for content in contents_path:
			result = _handler(content, model_path, save=save, postfix=postfix)
			outputs.append(np.squeeze(result, axis=0))
		return outputs


def _handler(content_path, model_path, resize_height=None, resize_width=None, save=True, postfix='-stylized'):
	# get the actual image data, output shape: (num_images, height, width, color_channels)
	content_target = get_images(content_path, resize_height, resize_width)

	with tf.Graph().as_default():
		# build the dataflow graph
		content_image = tf.placeholder(tf.float32, shape=content_target.shape, name='content_image')

		output_image = itn.transform(content_image / 255.0)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()

		with tf.Session() as sess:
			saver.restore(sess, model_path)
			output = sess.run(output_image, feed_dict={content_image: content_target})

	if save:
		save_images(content_path, output, postfix)

	return output

