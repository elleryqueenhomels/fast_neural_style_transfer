# Demo - train the style transfer network & use it to generate an image


from train import train
from generate import generate
from utils import list_images


STYLE = 'wave'
IS_TRAINING = True


if __name__ == '__main__':
    content_weight = 1.0
    style_weight   = 8.0
    tv_weight      = 0.01

    style_target = 'images/style/' + STYLE + '.jpg'
    model_save_path = 'models/' + STYLE + '.ckpt-done'

    if IS_TRAINING:
        content_targets = list_images('../MS_COCO')

        train(content_targets, style_target, content_weight, style_weight, tv_weight, save_path=model_save_path, debug=True)

        print('\nSuccessfully! Done training...\n')
    else:
        model_path = model_save_path
        output_save_path = 'outputs'

        content_targets = list_images('images/content')
        generated_images = generate(content_targets, model_path, save_path=output_save_path, postfix='-' + STYLE)

        print('\ntype(generated_images):', type(generated_images))
        print('\nlen(generated_images):', len(generated_images), '\n')

