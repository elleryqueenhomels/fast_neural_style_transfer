# Utility

import tensorflow as tf


def get_image(path, height=None, width=None):
	with tf.Graph().as_default(), tf.Session() as sess:

		img_format = path.lower()[-3:]
		img_bytes = tf.read_file(path)

		if img_format == 'png':
			image = tf.image.decode_png(img_bytes, channels=3)
		else:
			image = tf.image.decode_jpeg(img_bytes, channels=3)

		if height is not None and width is not None:
			image = tf.image.resize_images(image, [height, width])

		image = tf.stack([image])

		return image.eval()


def get_batch_images(paths, height=None, width=None):
	with tf.Graph().as_default(), tf.Session() as sess:

		images = []
		for path in paths:
			img_format = path.lower()[-3:]
			img_bytes = tf.read_file(path)

			if img_format == 'png':
				image = tf.image.decode_png(img_bytes, channels=3)
			else:
				image = tf.image.decode_jpeg(img_bytes, channels=3)
			
			images.append(image)

		images = tf.stack(images)

		if height is not None and width is not None:
			images = tf.image.resize_images(images, [height, width])

		return images.eval()

