import argparse
import cv2
import os

from torchvision.transforms import Compose
from transform import Resize


def main(args):

    transform = Compose([
        Resize(
            width=args.width,
            height=args.height,
            resize_target=False,
            keep_aspect_ratio=False,
            ensure_multiple_of=16,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_AREA
            # If you are enlarging the image, you should prefer to use INTER_LINEAR 
            # or INTER_CUBIC interpolation. If you are shrinking the image, you 
            # should prefer to use INTER_AREA interpolation. 
            # reference: https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
        )
    ])

    # assuming root dir has only images
    images = os.listdir(args.root_dir)  

    # create output directory 
    os.makedirs(args.out_dir, exist_ok=True)

    for image_path in images:
        image = cv2.imread(os.path.join(args.root_dir, image_path), cv2.IMREAD_UNCHANGED)
        sample = transform({'image': image})
        image = sample['image']

        # with .bmp extension
        # image_path = f'{os.path.splitext(image_path)[0]}x3{os.path.splitext(image_path)[1]}'
        # image_path = f'{os.path.splitext(image_path)[0]}{os.path.splitext(image_path)[1]}'

        # with .png extension, it weights less
        # image_path = f'{os.path.splitext(image_path)[0]}x3.png'
        image_path = f'{os.path.splitext(image_path)[0]}.png'

        cv2.imwrite(os.path.join(args.out_dir, image_path), image)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Downscale images for SR training.')
    parser.add_argument('--root_dir', required=True, help='Path to the dataset folder.')
    parser.add_argument('--out_dir', required=True, help='Folder target path.')
    parser.add_argument('--width', type=int, default=64, help='Width of the downscaled image.')
    parser.add_argument('--height', type=int, default=64, help='Height of the downscaled image.')
    args = parser.parse_args()
    main(args)
