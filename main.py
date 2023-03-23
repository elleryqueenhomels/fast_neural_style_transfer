# Demo - train the style transfer network & use it to generate an image

import argparse

from train import train
from generate import generate
from utils import list_images

parser = argparse.ArgumentParser()

parser.add_argument("--is_training", type=bool, default=False, help="Using training mode, default value is False")
parser.add_argument("--vgg_path", default="imagenet-vgg-19-weights.npz", help="VGG weights file path")

args = parser.parse_args()

VGG_PATH = args.vgg_path

# format: {'style': [content_weight, style_weight, tv_weight]}
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

def main():

    if args.is_training:

        content_targets = list_images('./MS_COCO') # path to training dataset

        for style in list(STYLES.keys()):

            print('\nBegin to train the network with the style "%s"...\n' % style)

            content_weight, style_weight, tv_weight = STYLES[style]

            style_target = 'images/style/' + style + '.jpg'
            model_save_path = 'models/' + style + '.ckpt-done'

            train(content_targets, style_target, content_weight, style_weight, tv_weight, 
                vgg_path=VGG_PATH, save_path=model_save_path, debug=True)

            print('\nSuccessfully! Done training style "%s"...\n' % style)

        print('Successfully finish all the training...\n')
    else:

        for style in list(STYLES.keys()):

            print('\nBegin to generate pictures with the style "%s"...\n' % style)

            model_path = 'models/' + style + '.ckpt-done'
            output_save_path = 'outputs'

            content_targets = list_images('images/content')
            generated_images = generate(content_targets, model_path, save_path=output_save_path, 
                prefix=style + '-')

            print('\ntype(generated_images):', type(generated_images))
            print('\nlen(generated_images):', len(generated_images), '\n')

if __name__ == '__main__':
    main()
