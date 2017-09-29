# Demo - train the style transfer network & use it to generate an image

from __future__ import print_function

from train import train
from generate import generate
from utils import list_images


IS_TRAINING = False


STYLES = {
    'wave':            [1.0,   7.0, 1e-2],
    'udnie':           [1.0,  12.0, 1e-2],
    'escher_sphere':   [1.0,  60.0, 1e-2],
    'flower':          [1.0,  10.0, 1e-2],
    'scream':          [1.0,  60.0, 1e-2],
    'denoised_starry': [1.0,  16.0, 1e-2],
    'starry_bright':   [1.0,   6.0, 1e-2],
    'rain_princess':   [1.0,   8.0, 1e-2],
    'woman_matisse':   [1.0,  20.0, 1e-2],
    'mosaic':          [1.0,   5.0,  0.0],
}


if __name__ == '__main__':

    if IS_TRAINING:

        content_targets = list_images('../MS_COCO') # path to training dataset

        for style in list(STYLES.keys()):

            print('\n\nBegin to train the network with the style "%s"...\n' % style)

            content_weight, style_weight, tv_weight = STYLES[style]

            style_target = 'images/style/' + style + '.jpg'
            model_save_path = 'models/' + style + '.ckpt-done'

            train(content_targets, style_target, content_weight, style_weight, tv_weight, save_path=model_save_path, debug=True)

            print('\nSuccessfully! Done training style "%s"...\n' % style)

        print('Successfully finish all the training...\n')
    else:

        for style in list(STYLES.keys()):

            print('\n\nBegin to generate pictures with the style "%s"...\n' % style)

            model_path = 'models/' + style + '.ckpt-done'
            output_save_path = 'outputs'

            content_targets = list_images('images/content')
            generated_images = generate(content_targets, model_path, save_path=output_save_path, prefix=style + '-')

            print('\ntype(generated_images):', type(generated_images))
            print('\nlen(generated_images):', len(generated_images), '\n')

