# Train the Image Transform Net using a fixed VGG19 as a Loss Network
# The VGG19 is pre-trained on ImageNet dataset

import numpy as np
import tensorflow as tf
import image_transform_net as itn

from loss_net import VGG, preprocess
from utils import get_image, get_batch_images


STYLE_LAYERS  = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'

TRAINING_IMAGE_SHAPE = (256, 256, 3) # (height, width, color_channels)

VGG_PATH = 'pretrained/imagenet-vgg-19-weights.npz'
SAVE_PATH = 'models/style-transfer-model.ckpt'


def train(content_targets_path, style_target_path, content_weight, style_weight, tv_weight, vgg_path=VGG_PATH, save_path=SAVE_PATH, epochs=2, batch_size=4, learning_rate=1e-3, debug=False, logging_period=1000):
	# guarantee the size of content_targets is a multiple of batch_size
	mod = len(content_targets_path) % batch_size
	if mod > 0:
		print('Train set has been trimmed %d samples...' % mod)
		content_targets_path = content_targets_path[:-mod]

	# retrive the style_target image
	style_target = get_image(style_target_path) # shape: (1, height, width, channels)

	# create a pre-trained VGG network
	vgg = VGG(vgg_path)

	input_shape = (batch_size, *TRAINING_IMAGE_SHAPE)
	style_shape = style_target.shape

	# compute the style features
	style_features = {}
	with tf.Graph().as_default(), tf.Session() as sess:
		style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')

		# pass style_image through 'pretrained VGG-19 network'
		style_img_preprocess = preprocess(style_image)
		style_net = vgg.forward(style_img_preprocess)

		for style_layer in STYLE_LAYERS:
			features = style_net[style_layer].eval(feed_dict={style_image: style_target})
			features = np.reshape(features, [-1, features.shape[3]])
			gram = np.matmul(features.T, features) / features.size
			style_features[style_layer] = gram

	# compute the perceptual losses
	with tf.Graph().as_default(), tf.Session() as sess:
		content_images = tf.placeholder(tf.float32, shape=input_shape, name='content_images')

		# pass content_images through 'pretrained VGG-19 network'
		content_imgs_preprocess = preprocess(content_images)
		content_net = vgg.forward(content_imgs_preprocess)

		# compute the content features
		content_features = {}
		content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

		# pass content_images through 'Image Transform Net'
		output_images = itn.transform(content_images / 255.0)

		# pass output_images through 'pretrained VGG-19 network'
		output_imgs_preprocess = preprocess(output_images)
		output_net = vgg.forward(output_imgs_preprocess)

		# ** compute the feature reconstruction loss **
		content_size = tf.size(content_features[CONTENT_LAYER])

		content_loss = content_weight * 2 * tf.nn.l2_loss(output_net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / tf.to_float(content_size)

		# ** compute the style reconstruction loss **
		style_losses = []
		for style_layer in STYLE_LAYERS:
			features = output_net[style_layer]
			shape = tf.shape(features)
			num_images, height, width, num_filters = shape[0], shape[1], shape[2], shape[3]
			features = tf.reshape(features, [num_images, height*width, num_filters])
			grams = tf.matmul(features, features, transpose_a=True) / tf.to_float(height * width * num_filters)
			style_gram = style_features[style_layer]
			layer_style_loss = 2 * tf.nn.l2_loss(grams - style_gram) / tf.to_float(tf.size(grams))
			style_losses.append(layer_style_loss)

		style_loss = style_weight * tf.reduce_sum(tf.stack(style_losses))

		# ** compute the total variation loss **
		shape = tf.shape(output_images)
		height, width = shape[1], shape[2]
		y = tf.slice(output_images, [0, 0, 0, 0], [-1, height - 1, -1, -1]) - tf.slice(output_images, [0, 1, 0, 0], [-1, -1, -1, -1])
		x = tf.slice(output_images, [0, 0, 0, 0], [-1, -1, width - 1, -1]) - tf.slice(output_images, [0, 0, 1, 0], [-1, -1, -1, -1])
		raw_tv_loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

		tv_loss = tv_weight * raw_tv_loss

		# overall perceptual losses
		loss = content_loss + style_loss + tv_loss

		# Training step
		train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

		sess.run(tf.global_variables_initializer())

		# saver = tf.train.Saver()
		saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

		# ** Start Training **
		step = 0
		n_batches = len(content_targets_path) // batch_size

		if debug:
			import time
			start_time = time.time()
			tf.logging.info('Start training the model...')

		for epoch in range(epochs):
			for batch in range(n_batches):
				# retrive a batch of content_targets images
				content_batch = content_targets_path[batch*batch_size:(batch*batch_size + batch_size)]
				content_batch = get_batch_images(content_batch, input_shape[1], input_shape[2])

				# run the training step
				sess.run(train_op, feed_dict={content_images: content_batch})

				step += 1

				if step % 1000 == 0:
					saver.save(sess, save_path, global_step=step)

				if debug:
					if step % logging_period == 0:
						elapsed_time = time.time() - start_time
						_content_loss, _style_loss, _tv_loss, _loss = sess.run([content_loss, style_loss, tv_loss, loss], feed_dict={content_images: content_batch})
						tf.logging.info('step: %d,  total loss: %f,  elapsed time: %f' % (step, _loss, elapsed_time))
						tf.logging.info('content loss: %f,  weighted content loss: %f' % (_content_loss, content_weight * _content_loss))
						tf.logging.info('style loss  : %f,  weighted style loss  : %f' % (_style_loss, style_weight * _style_loss))
						tf.logging.info('tv loss     : %f,  weighted tv loss     : %f' % (_tv_loss, tv_weight * _tv_loss))

		# ** Done Training & Save the model **
		saver.save(sess, save_path)

