# Demo - train the style transfer network & use it to generate an image

from train import train
from generate import generate


if __name__ == '__main__':
	content_weight = 1.0
	style_weight = 220.0
	tv_weight = 0.1

	content_targets = ['images/content/lena.png', 'images/content/lena2.png']
	style_target = 'images/style/starry.jpg'

	SAVE_PATH = 'models/wave.ckpt'

	# train(content_targets, style_target, content_weight, style_weight, tv_weight, save_path=SAVE_PATH, batch_size=2, debug=True)

	# print('\nSuccessfully! Done training...\n')

	# content_targets = ['images/content/lena.png', 'images/content/lena2.png', 'images/content/lena3.png', 'images/content/lena4.png', 'images/content/lena5.png',]
	# content_targets = ['images/content/cubist.jpg']
	content_targets = 'images/content/scream.jpg'

	model_path = 'models/wave.ckpt'

	generated_images = generate(content_targets, model_path)

	print('\ntype(generated_images):', type(generated_images))
	print('generated_images.shape:', generated_images.shape, '\n')

	import matplotlib.pyplot as plt

	for img in generated_images:
		plt.imshow(img)
		plt.show()

