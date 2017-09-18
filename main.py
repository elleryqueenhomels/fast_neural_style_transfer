# Demo - train the style transfer network & use it to generate an image


from __future__ import print_function

from train import train
from generate import generate
from utils import list_images


IS_TRAINING = True

STYLES = ['wave', 'udnie', 'starry', 'scream', 'denoised_starry', 'mosaic']

CONTENT_WEIGHTS = [ 1.0,   1.0,  1.0,  1.0,  1.0,   1.0]
STYLE_WEIGHTS   = [10.0, 100.0, 15.0, 15.0, 15.0, 100.0]
TV_WEIGHTS      = [0.01,  0.01, 0.01, 0.01, 0.01,  0.01]


if __name__ == '__main__':

    if IS_TRAINING:

        content_targets = list_images('../MS_COCO') # path to training dataset

        for style, content_weight, style_weight, tv_weight in zip(STYLES, CONTENT_WEIGHTS, STYLE_WEIGHTS, TV_WEIGHTS):

            print('\n\nBegin to train the network with the style "%s"...\n' % style)

            style_target = 'images/style/' + style + '.jpg'
            model_save_path = 'models/' + style + '.ckpt-done'

            train(content_targets, style_target, content_weight, style_weight, tv_weight, save_path=model_save_path, debug=True)

            print('\nSuccessfully! Done training...\n')
    else:

        for style in STYLES:

            print('\n\nBegin to generate pictures with the style "%s"...\n' % style)

            model_path = 'models/' + style + '.ckpt-done'
            output_save_path = 'outputs'

            content_targets = list_images('images/content')
            generated_images = generate(content_targets, model_path, save_path=output_save_path, postfix='-' + style)

            print('\ntype(generated_images):', type(generated_images))
            print('\nlen(generated_images):', len(generated_images), '\n')

