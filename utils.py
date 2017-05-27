# Utility

import tensorflow as tf

from os import listdir, mkdir, sep
from os.path import join, exists, splitext


def list_images(directory):
	images = []
	for file in listdir(directory):
		name = file.lower()
		if name.endswith('.png'):
			images.append(join(directory, file))
		elif name.endswith('.jpg'):
			images.append(join(directory, file))
		elif name.endswith('.jpeg'):
			images.append(join(directory, file))
	return images


def get_images(paths, height=None, width=None):
	if isinstance(paths, str):
		paths = [paths]

	with tf.Graph().as_default(), tf.Session() as sess:

		images = []
		for path in paths:
			is_png = path.lower().endswith('png')
			img_bytes = tf.read_file(path)

			if is_png:
				image = tf.image.decode_png(img_bytes, channels=3)
			else:
				image = tf.image.decode_jpeg(img_bytes, channels=3)

			if height is not None and width is not None:
				image = tf.image.resize_images(image, [height, width])
			
			images.append(image)

		images = tf.stack(images)

		return images.eval()


def save_images(paths, datas, save_path, postfix='-stylized'):
	if isinstance(paths, str):
		paths = [paths]

	assert(len(paths) == len(datas))

	if not exists(save_path):
		mkdir(save_path)

	with tf.Graph().as_default(), tf.Session() as sess:

		for i, path in enumerate(paths):
			data = datas[i]
			is_png = path.lower().endswith('png')

			if is_png:
				image = tf.image.encode_png(data)
			else:
				image = tf.image.encode_jpeg(data)

			name, ext = splitext(path)
			name = name.split(sep)[-1]
			if postfix is not None:
				path = join(save_path, name + postfix + ext)
			else:
				path = join(save_path, name + ext)

			tf.write_file(path, image).run()

