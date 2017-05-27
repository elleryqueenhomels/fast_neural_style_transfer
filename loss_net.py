# Part of VGG19 - Deep Convolutional Neural Network
# We only make use of the ConvPoolLayer,
# discard the fully-connected hidden layer.
# Pre-trained on ImageNet dataset
# Used for defining and computing:
# feature reconstruction loss & style reconstruction loss

import numpy as np
import tensorflow as tf


MEAN_PIXEL = np.array([123.68, 116.779, 103.939]) # RGB

VGG19_LAYERS = (
	'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

	'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

	'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
	'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

	'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
	'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

	'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
	'relu5_3', 'conv5_4', 'relu5_4'
)


class VGG(object):
	def __init__(self, weights_path, layers=VGG19_LAYERS):
		self.weights = load_vgg_weights(weights_path)
		self.layers = layers

	def forward(self, image):
		idx = 0
		net = {}
		current = image
		for name in self.layers:
			kind = name[:4]

			if kind == 'conv':
				kernel, bias = self.weights[idx]
				idx += 1
				current = conv_layer(current, kernel, bias)
			elif kind == 'relu':
				current = tf.nn.relu(current)
			elif kind == 'pool':
				current = pool_layer(current)

			net[name] = current

		assert(len(net) == len(self.layers))
		return net


def conv_layer(x, weight, bias):
	conv = tf.nn.conv2d(x, tf.constant(weight), strides=[1, 1, 1, 1], padding='SAME')
	return tf.nn.bias_add(conv, bias)


def pool_layer(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def load_vgg_weights(weights_path):
	kind = weights_path[-3:]
	if kind == 'npz':
		weights = load_from_npz(weights_path)
	elif kind == 'mat':
		weights = load_from_mat(weights_path)
	else:
		weights = None
		print('Unrecognized file type: %s' % kind)
	return weights


def load_from_npz(weights_path):
	params = np.load(weights_path)
	count = int(params['arr_0']) + 1
	weights = []
	for i in range(1, count, 2):
		kernel = params['arr_%s' % i]
		bias = params['arr_%s' % (i + 1)]
		weights.append((kernel, bias))
	return weights


def load_from_mat(weights_path):
	from scipy.io import loadmat
	data = loadmat(weights_path)
	if not all(i in data for i in ('layers', 'classes', 'normalization')):
		raise ValueError('You are using the wrong VGG-19 data.')
	params = data['layers'][0]

	weights = []
	for i, name in enumerate(VGG19_LAYERS):
		if name[:4] == 'conv':
			# matlabconv: [width, height, in_channels, out_channels]
			# tensorflow: [height, width, in_channels, out_channels]
			kernel, bias = params[i][0][0][0][0]
			kernel = np.transpose(kernel, [1, 0, 2, 3])
			bias = bias.reshape(-1) # flatten
			weights.append((kernel, bias))
	return weights


def preprocess(image, mean=MEAN_PIXEL):
	return image - mean


def unprocess(image, mean=MEAN_PIXEL):
	return image + mean


# test
if __name__ == '__main__':
	weights_path = 'pretrained/imagenet-vgg-19-weights.npz'
	image = tf.placeholder(tf.float32, shape=(None, 256, 256, 3), name='x')

	vgg = VGG(weights_path)
	net = vgg.forward(image)

	print('\nSuccessfully!\n')

