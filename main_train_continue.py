import argparse

import lpips
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from data.select_dataset import define_Dataset
from models.select_model import define_Model
from utils import *
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main(
        json_path='',
):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', default=0, help='local rank')
    parser.add_argument('--dist', default=False, help='dist')
    parser.add_argument('--idx', default=-1, type=int, help='compare idx')
    args = parser.parse_args()
    opt = parse(args.opt, is_train=True)
    if parser.parse_args().dist is False:
        opt['dist'] = False
        print('not use dist')
    else:
        print('use dist')
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))
        lpips_ = lpips.LPIPS(net='alex')

    init_iter_G, init_path_G = find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = find_last_checkpoint(opt['path']['models'], net_type='R')


    if init_path_G != None:
        opt['path']['pretrained_netG'] = init_path_G
        opt['train']['G_param_strict'] = True
    if init_path_E != None:
        opt['path']['pretrained_netE'] = init_path_E
        opt['train']['E_param_strict'] = True

    init_iter_optimizerG, init_path_optimizerG = find_last_checkpoint(opt['path']['models'],
                                                                      net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG, 0)
    border = opt['scale']

    if opt['rank'] == 0:
        save(opt)

    opt = dict_to_nonedict(opt)

    if opt['rank'] == 0:
        logger_name = 'train'
        logger = utils_logger.get_logger(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
        logger.info(os.path.basename(__file__))

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = 3407
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = define_Model(opt, phase='train')
    model.init_train()
    tile = opt['tile']
    print('tile:', tile)

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))

            opt['train']['train_size'] = train_size
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
        elif phase == 'eval':
            eval_set = define_Dataset(dataset_opt)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    batchsize = opt['datasets']['train']['dataloader_batch_size']
    print('batchsize', batchsize)
    all_tasks = train_set.tasks
    for task_id in range(all_tasks):
        if task_id==0 :
            continue
        if current_step >= opt['train']["G_scheduler_milestones"][task_id][-1]:
            continue
        model.resume(current_step=current_step, task_id=task_id)

        train_set.load_dataset(task_id)
        if opt['dist']:
            train_sampler = DistributedSampler(train_set,
                                               shuffle=opt['datasets']['train']['dataloader_shuffle'], drop_last=True,
                                               seed=seed)
            train_loader = DataLoader(train_set,
                                      batch_size=opt['datasets']['train']['dataloader_batch_size'] // opt['num_gpu'],
                                      shuffle=False,
                                      num_workers=opt['datasets']['train']['dataloader_num_workers'] // opt['num_gpu'],
                                      drop_last=True,
                                      pin_memory=True,
                                      sampler=train_sampler, collate_fn=collate_fn)
        else:
            train_loader = DataLoader(train_set,
                                      batch_size=opt['datasets']['train']['dataloader_batch_size'],
                                      shuffle=opt['datasets']['train']['dataloader_shuffle'],
                                      num_workers=opt['datasets']['train']['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True,
                                      collate_fn=collate_fn,
                                      )
        epoch = 0
        print("now step", current_step, "task max step", opt['train']["G_scheduler_milestones"][task_id][-1])
        stop = False
        while not stop:
            if opt['dist']:
                train_sampler.set_epoch(epoch)
            bar = tqdm(train_loader)
            for i, train_data in enumerate(bar):
                current_step += batchsize
                if current_step > opt['train']["G_scheduler_milestones"][task_id][-1]:
                    stop = True
                if eval(opt['train']['checkpoint_test'][task_id]) != 1:
                    model.update_learning_rate(current_step)
                    model.feed_data(train_data)
                    model.optimize_parameters(current_step)


                if current_step % eval(opt['train']['checkpoint_print']) == 0 and opt['rank'] == 0:
                    logs = model.current_log()
                    message = '<Task:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(task_id, int(current_step),
                                                                             model.current_learning_rate())
                    for k, v in logs.items():
                        message += '{:s}: {:.3e} '.format(k, v)
                    logger.info(message)
                if current_step % eval(opt['train']['checkpoint_save'][task_id]) == 0 and opt['rank'] == 0:
                    logger.info('Saving the models.')
                    model.save(int(current_step))

                if (current_step % eval(opt['train']['checkpoint_test'][task_id]) == 0 or stop) and opt['rank'] == 0:
                    model.netG.eval()
                    model.save(int(current_step), save_best=True)
                    if current_step < 0:
                        val_set = eval_set
                    else:
                        val_set = test_set
                    tasks = val_set.tasks
                    all_avg_psnr_y = 0
                    all_avg_ssim_y = 0
                    if stop:
                        start = 0
                    else:
                        start = task_id
                    for id in range(start, min(task_id, tasks) + 1):
                        avg_psnr = 0.0
                        avg_ssim = 0.0
                        avg_ssim_y = 0.0
                        avg_lpips = 0.0
                        avg_psnr_y = 0.0
                        avg_niqe_y = 0.0
                        test_iter = 0
                        val_set.load_dataset(id)
                        loader = DataLoader(val_set, batch_size=1,
                                            shuffle=False, num_workers=1,
                                            drop_last=False, pin_memory=True)
                        for test_data in loader:
                            test_iter += 1
                            image_name_ext = os.path.basename(test_data['path'][0])
                            model.feed_data(test_data)

                            _, _, h_old, w_old = model.L[0].size()
                            lr = model.L

                            with torch.no_grad():
                                E_img = model.ttest(lr, scale=opt['scale'], tile=tile)
                                while type(E_img) is list:
                                    E_img = E_img[args.idx]
                            H_img = tensor2uint(model.H[0].detach().float().cpu())
                            E_img = tensor2uint(E_img[:, :, :h_old * opt['scale'], :w_old * opt['scale']])

                            current_psnr = calculate_psnr(E_img, H_img, border=opt['scale'], maxn=255)
                            current_lpips = calculate_lpips(E_img / 255, H_img / 255, lpips_=lpips_, border=border,
                                                            maxn=1)
                            current_ssim = calculate_ssim(E_img, H_img, border=opt['scale'], maxn=255)
                            E_img0 = rgb2ycbcr(E_img.astype(np.float32) / 255.) * 255.
                            H_img0 = rgb2ycbcr(H_img.astype(np.float32) / 255.) * 255.
                            current_ssim_y = calculate_ssim(E_img0, H_img0, border=opt['scale'], maxn=255)
                            current_psnr_y = calculate_psnr(E_img0, H_img0, border=border, maxn=255)
                            current_niqe_y = calculate_niqe(E_img0, border=opt['scale'])

                            if stop:
                                logger.info(
                                    '{:->4d}--> {:>10s} | {:<4.2f} | {:<4.3f} | {:<4.3f} | {:<4.3f} | {:<4.3f}'.format(
                                        test_iter,
                                        image_name_ext,
                                        current_psnr_y,
                                        current_ssim,
                                        current_ssim_y,
                                        current_lpips,
                                        current_niqe_y
                                    ))
                            else:
                                print(
                                    '{:->4d}--> {:>10s} | {:<4.2f} | {:<4.3f} | {:<4.3f} | {:<4.3f} | {:<4.3f}'.format(
                                        test_iter,
                                        image_name_ext,
                                        current_psnr_y,
                                        current_ssim,
                                        current_ssim_y,
                                        current_lpips,
                                        current_niqe_y
                                    ))
                            avg_psnr += current_psnr
                            avg_ssim_y += current_ssim_y
                            avg_ssim += current_ssim
                            avg_lpips += current_lpips
                            avg_psnr_y += current_psnr_y
                            avg_niqe_y += current_niqe_y
                        avg_psnr = avg_psnr / test_iter
                        avg_ssim_y = avg_ssim_y / test_iter
                        avg_ssim = avg_ssim / test_iter
                        avg_lpips = avg_lpips / test_iter
                        avg_psnr_y = avg_psnr_y / test_iter
                        avg_niqe_y = avg_niqe_y / test_iter

                        logger.info(
                            '<iter:{:8,d}, Task:{:3d} Average PSNR|SSIMY|SSIM|LPIPS|NIQE: {:<.2f} | {:<.3f} | {:<.3f}| {:<.3f}| {:<.3f}\n'
                                .format(int(current_step), id, avg_psnr_y, avg_ssim_y, avg_ssim, avg_lpips,
                                        avg_niqe_y))
                        all_avg_psnr_y += avg_psnr_y
                        all_avg_ssim_y += avg_ssim_y
                    all_avg_psnr_y = all_avg_psnr_y / (task_id - start + 1)
                    all_avg_ssim_y = all_avg_ssim_y / (task_id - start + 1)

                    logger.info(
                        '<iter:{:8,d}, All Average PSNR|SSIMY: {:<.2f} | {:<.3f} \n'
                            .format(int(current_step), all_avg_psnr_y, all_avg_ssim_y))

                    model.netG.train()
                    if stop:
                        break
            epoch += 1


if __name__ == '__main__':
    main()
