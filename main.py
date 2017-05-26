# Demo - train the style transfer network & use it to generate an image

from train import train
from generate import generate
from utils import list_images


is_training = False
condition = 1

if __name__ == '__main__':
	content_weight = 1.0
	style_weight = 220.0
	tv_weight = 0.1

	style_target = 'images/style/starry.jpg'
	save_path = 'models/starry.ckpt-done'

	if is_training:
		content_targets = list_images('images/content')
		print('\ncontent_targets:\n', content_targets, '\n')

		train(content_targets, style_target, content_weight, style_weight, tv_weight, save_path=save_path, batch_size=2, debug=True)
		print('\nSuccessfully! Done training...\n')
	else:
		model_path = save_path

		if condition == 1:
			content_targets = list_images('images/content')
			generated_images = generate(content_targets, model_path, resize_height=300, resize_width=300, postfix='-starry')
		elif condition == 2:
			content_targets = list_images('images/content')
			generated_images = generate(content_targets, model_path, postfix='-starry')
		elif condition == 3:
			content_targets = ['images/content/lena.png', 'images/content/lena2.png', 'images/content/lena3.png', 'images/content/lena4.png']
			generated_images = generate(content_targets, model_path, is_same_size=True, postfix='-starry')
		else:
			content_targets = 'images/content/scream.jpg'
			generated_images = generate(content_targets, model_path, postfix='-starry')

		print('\ntype(generated_images):', type(generated_images))
		print('\nlen(generated_images):', len(generated_images), '\n')

		import matplotlib.pyplot as plt

		for img in generated_images:
			plt.imshow(img)
			plt.show()

