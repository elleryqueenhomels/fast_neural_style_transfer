# Demo - train the style transfer network & use it to generate an image


from train import train
from generate import generate
from utils import list_images


is_training = True
condition = 1

if __name__ == '__main__':
    content_weight = 1.0
    style_weight = 80.0
    tv_weight = 0.01

    style_target = 'images/style/wave.jpg'
    model_save_path = 'models/wave.ckpt-done'

    if is_training:
        content_targets = list_images('../MS_COCO')

        train(content_targets, style_target, content_weight, style_weight, tv_weight, save_path=model_save_path, debug=True)
        print('\nSuccessfully! Done training...\n')
    else:
        model_path = model_save_path
        output_save_path = 'outputs'

        if condition == 1:
            content_targets = list_images('images/content')
            generated_images = generate(content_targets, model_path, save_path=output_save_path, postfix='-starry')
        elif condition == 2:
            content_targets = list_images('images/content')
            generated_images = generate(content_targets, model_path, save_path=output_save_path, postfix='-starry')
        elif condition == 3:
            content_targets = ['images/content/lena.png', 'images/content/lena2.png', 'images/content/lena3.png', 'images/content/lena4.png']
            generated_images = generate(content_targets, model_path, is_same_size=True, save_path=output_save_path, postfix='-starry')
        else:
            content_targets = 'images/content/scream.jpg'
            generated_images = generate(content_targets, model_path, save_path=output_save_path, postfix='-starry')

        print('\ntype(generated_images):', type(generated_images))
        print('\nlen(generated_images):', len(generated_images), '\n')
        

        # import matplotlib.pyplot as plt

        # for img in generated_images:
        #     plt.imshow(img)
        #     plt.show()

