# Use a trained Image Transform Net to generate
# a style transferred image with a specific style

import tensorflow as tf
import image_transform_net as itn

from utils import get_images, save_images


def generate(contents_path, model_path, is_same_size=False, resize_height=None, resize_width=None, save_path=None, postfix='-stylized'):
	if isinstance(contents_path, str):
		contents_path = [contents_path]

	if is_same_size or (resize_height is not None and resize_width is not None):
		outputs = _handler1(contents_path, model_path, resize_height=resize_height, resize_width=resize_width, save_path=save_path, postfix=postfix)
		return [outputs[i] for i in range(len(outputs))]
	else:
		outputs = _handler2(contents_path, model_path, save_path=save_path, postfix=postfix)
		return outputs


def _handler1(content_path, model_path, resize_height=None, resize_width=None, save_path=None, postfix='-stylized'):
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

	if save_path is not None:
		save_images(content_path, output, save_path, postfix)

	return output


def _handler2(content_path, model_path, save_path=None, postfix='-stylized'):
	with tf.Graph().as_default():
		# build the dataflow graph
		content_image = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='content_image')

		output_image = itn.transform(content_image / 255.0)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()

		with tf.Session() as sess:
			saver.restore(sess, model_path)

			output = []
			for content in content_path:
				content_target = get_images(content)
				result = sess.run(output_image, feed_dict={content_image: content_target})
				output.append(result[0])

	if save_path is not None:
		save_images(content_path, output, save_path, postfix)

	return output

