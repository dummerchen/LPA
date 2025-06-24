import argparse
import os
import random
from collections import OrderedDict

import lpips
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.select_dataset import define_Dataset
from models import define_Model
from utils.utils_image import mkdir, tensor2uint, imsave, rgb2ycbcr, calculate_ssim, \
    calculate_psnr, calculate_lpips
from utils.utils_logger import get_logger
from utils.utils_model import find_best_checkpoint
from utils.utils_niqe import calculate_niqe
from utils.utils_option import parse


def test(opt: OrderedDict, args):
    opt['datasets']['test']['scale'] = opt['scale']
    opt['datasets']['test']['n_channels'] = opt['n_channels']
    opt['datasets']['test']['phase'] = 'test'
    opt['datasets']['test']['H_size'] = None
    opt['train']['G_param_strict'] = True
    opt['dist'] = False
    opt['is_train'] = False

    seed = 3407
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    epoch, opt['path']['pretrained_netG'] = find_best_checkpoint(opt['path']['models'], net_type='G', epoch=args.epoch)
    logger = get_logger('test', os.path.join(os.path.dirname(opt['path']['pretrained_netG']), '..', 'test.log'))

    logger.info(opt['file_name'])
    logger.info("tile: {}".format(args.tile))
    logger.info("pretrain weight: {}".format(opt['path']['pretrained_netG']))
    logger.info("dataset: {}".format(opt['datasets']['test']['dataroot_L']))
    # must use train to init norm layer
    # model = define_Model(opt, phase='train')
    model = define_Model(opt, phase='test')
    model.netG.to(args.device)
    flag = model.opt.get('flag')
    logger.info("flag:{}".format(flag))

    model.load()
    try:
        # 初始化ours 的task identifier
        model.netG.prompt.process_task_add(task_id=args.n_tasks - 1, flag=flag)
        model_name = args.model_name if args.model_name is not None else opt['netG']['prompt'].get('net_type')
        print('model_name:{}'.format(model_name))
        if 'codap' not in model_name.lower():
            model.netG.prompt.task_id = None
        print('now task id is:', model.netG.prompt.task_id)
        logger.info("init model task id is None")
    except Exception as e:
        print('Exception:', e)
        model_name = 'none'
    opt['datasets']['test']['dataset_type'] = 'ContinueSR'
    test_set = define_Dataset(opt['datasets']['test'])
    res_psnr_y = []
    res_ssim_y = []
    dataroot_L = opt['datasets']['test']['dataroot_L']
    start_task = args.start_task if args.start_task is not None else 0
    for t in range(start_task, min(args.n_tasks, test_set.tasks)):
        test_set.load_dataset(t)
        print('Start Testdataset: {}'.format(opt['datasets']['test']['dataroot_L'][t]))
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
        lpips_ = lpips.LPIPS(net='alex').eval()
        avg_psnr = []
        avg_ssim = []
        avg_ssim_y = []
        avg_niqe_y = []
        avg_psnr_y = []
        avg_lpips = []
        avg_acc = []
        window_size = None
        model.netG.eval()
        # model.netG.prompt.task_id = t
        print('test loader data length | window size:', len(test_loader), window_size)
        specific = dataroot_L[t].split('/')[-3]+'/'+dataroot_L[t].split('/')[-2]
        for idx, data in enumerate(test_loader):
            image_name_ext = os.path.basename(data['path'][0])
            img_name, ext = os.path.splitext(image_name_ext)
            img_dir = os.path.join(os.path.dirname(opt['path']['pretrained_netG']),
                                   '../images/continue/{}/{}/{}'.format(str(opt['scale']), str(args.tile),
                                                                           str(epoch)), specific, img_name)
            mkdir(img_dir)
            with torch.no_grad():
                img_lq = [i.float().to(args.device) for i in data['L']]
                _, _, h_old, w_old = img_lq[0].shape

                E = model.ttest(img_lq, opt['scale'], tile=args.tile, task_id=None)
                while type(E) is list:
                    E = E[args.idx]
                H = data['H'][0].to(args.device)

                E_img = tensor2uint(E[:, :, :h_old * opt['scale'], :w_old * opt['scale']])
                H_img = tensor2uint(H)
                if args.save:
                    imsave(E_img, img_path=os.path.join(img_dir, img_name + '_{}.png'.format(model_name)))
                    imsave(H_img, img_path=os.path.join(img_dir, img_name + '_gt.png'))
                # -----------------------
                # calculate PSNRY
                # -----------------------
                current_psnr = calculate_psnr(E_img, H_img, border=opt['scale'], maxn=255)
                current_ssim = calculate_ssim(E_img, H_img, border=opt['scale'], maxn=255)
                current_lpips = calculate_lpips(E_img / 255, H_img / 255, lpips_, border=opt['scale'], maxn=1)
                E_img0 = rgb2ycbcr(E_img.astype(np.float32) / 255.) * 255.
                H_img0 = rgb2ycbcr(H_img.astype(np.float32) / 255.) * 255.
                current_ssim_y = calculate_ssim(E_img0, H_img0, border=opt['scale'], maxn=255)
                current_niqe_y = calculate_niqe(E_img0, border=opt['scale'])
                current_psnr_y = calculate_psnr(E_img0, H_img0, border=opt['scale'], maxn=255)
                try:
                    if type(model.netG).__name__ in ['DataParallel', 'DistributedDataParallel']:
                        current_task_id = model.netG.module.prompt.current_task_id
                    else:
                        current_task_id = model.netG.prompt.current_task_id
                except Exception as e:
                    current_task_id = -1
                    # print('Exception:', e, 'wo current task id')
            model.logger.info(
                '{:->4d}--> {:>10s} | {:<4.2f} | {:<4.3f} | {:<4.3f} | {:<4.3f} | {:<4.3f}'.format(idx,
                                                                                                   image_name_ext,
                                                                                                   current_psnr_y,
                                                                                                   current_ssim_y,
                                                                                                   current_ssim,
                                                                                                   current_task_id,
                                                                                                   current_lpips))
            avg_psnr.append(current_psnr)
            avg_acc.append(1 if current_task_id == t else 0)
            avg_ssim.append(current_ssim)
            avg_ssim_y.append(current_ssim_y)
            avg_psnr_y.append(current_psnr_y)
            avg_niqe_y.append(current_niqe_y)
            avg_lpips.append(current_lpips)
        model.logger.info(
            '-- Average PSNRY|SSIMY|SSIM|ACC|LPIPS: {:<4.2f} | {:<4.3f} | {:<4.3f} | {:<4.3f} | {:<4.3f}'.format(
                np.mean(avg_psnr_y),
                np.mean(avg_ssim_y),
                np.mean(avg_ssim),
                np.mean(avg_acc),
                np.mean(avg_lpips)))
        res_ssim_y.append(np.mean(avg_ssim_y))
        res_psnr_y.append(np.mean(avg_psnr_y))
    logger.info(
        'ALL Average PSNRY|SSIMY: {:<4.2f}|{:<4.3f}'.format(np.mean(res_psnr_y), np.mean(res_ssim_y)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str,
                        # default='options/swinir/finetune/x4/train_continue_finetune_swinir_d180_s4_t32_nrdem_brcmj.json'
                        # default='/home/dummerfu/clpsr/code/options/swinir/continue/x4/train_continue_freezeall_codap_ps[20]_pl[8]_bswinir101_p012345_fclip_d180_s4_t32_nrdem_brcmj.json'
                        # default='options/swinir/continue/x4/train_continue_freezeall_l2p_ps[20]_pl[8]_bswinir101_fclip_d180_s4_t32_nrdem_brcmj.json'
                        # default='options/swinir/continue/x4/train_continue_freezeall_codap_ps[20]_pl[8]_bswinir101_p012345_fclip_d180_s4_t32_nrdem_brcmj.json'
                        default='options/swinir/abl/module/train_continue_freezeall_promptv6_1_2_3_3_2_ps[3]_pl[12]_bswinir500_p012345_fmsr_clip_d180_s4_t32_nrdem_brcmj.json'
                        # default='options/swinir/x4/pswinir_d180_s4_t48.json'
                        # default='options/swinir/x4/train_finetune_swinir_d180_s4_t32_nrdi_uuuu_ub.json'
                        # default='options/dat/pdat_d180_s4_t64.json'
                        # default='options/bsrgan/x4/train_bsr_psnr_x4_wg_nrdi.json'
                        )
    parser.add_argument('-d', '--device', type=str, default='cuda:2')
    parser.add_argument('-mn', '--model_name', type=str, help='The suffix of the saved image file name')
    parser.add_argument('-t', '--tile', type=int, default=96, help='Tile size, 0 for no tile during testing')
    parser.add_argument('-nt', '--n_tasks', type=int, default=5, help='Total tasks default is 5')
    parser.add_argument('-i', '--idx', type=int, default=-1)
    parser.add_argument('-st', '--start_task', type=int, default=1, help='default start task is 0')
    parser.add_argument('-e', '--epoch', type=int, default=None, help='Specify the weight, the default is the last weight')
    parser.add_argument('--save', type=bool, default=False, help='save images')

    args = parser.parse_args()
    if args.tile == 0:
        args.tile = None

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device[-1]
    args.device = 'cuda:0'
    print('use single device:', args.device)
    with open(args.opt, 'r', encoding='utf-8') as f:
        json_str = f.read()
    opts = parse(args.opt, is_train=False)
    opts['file_name'] = os.path.basename(__file__)
    scale = opts['scale']
    test(opt=opts, args=args)
