import logging
import torch
from os import path as osp
import sys
import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio 

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from basicsr.data import build_dataloader, build_dataset
from basicsr.utils import (
    get_env_info, get_root_logger, get_time_str, make_exp_dirs
    )
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')

        total_psnr, total_ssim = 0, 0

        for sample in test_loader:
            ground_truth = sample['gt'].squeeze().permute(1, 2, 0).numpy()
            upsampled = sample['lq'].squeeze().permute(1, 2, 0).numpy()
            upsampled = cv2.resize(upsampled, ground_truth.shape[:2], interpolation=cv2.INTER_CUBIC)

            psnr = peak_signal_noise_ratio(ground_truth, upsampled)
            score = structural_similarity(upsampled, ground_truth, 
                                          channel_axis=2, data_range=1)
            total_ssim += score
            total_psnr += psnr

            # for visualization we have to reescale the image
            upsampled = np.clip(upsampled, 0., 1.) * 255

            save_path = f"{osp.splitext(osp.basename(sample['gt_path'][0]))[0]}_bicubic{osp.splitext(osp.basename(sample['gt_path'][0]))[1]}"
            cv2.imwrite(osp.join(opt['path']['visualization'], save_path), upsampled.astype('uint8'))

        logger.info(f'Mean PSNR {total_psnr/len(test_loader)}')
        logger.info(f'Mean SSIM {total_ssim/len(test_loader)}')
            

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)