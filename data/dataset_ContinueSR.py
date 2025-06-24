import torch.utils.data as data
from utils import *

class DatasetContinueSR(data.Dataset):
    def __init__(self, opt):
        super(DatasetContinueSR, self).__init__()
        self.opt = opt
        self.phase = opt['phase']
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.paths_L = get_image_paths(opt['dataroot_L'], get_image=False)
        self.paths_H = get_image_paths(opt['dataroot_H'], get_image=False)
        self.tasks = len(opt['dataroot_H'])
        assert self.paths_H, 'Error: H path is empty.'
        print('All {} HR|LR images {}|{}'.format(self.phase, len(self.paths_H), len(self.paths_L)))
        patch_size = self.opt.get('H_size')
        if patch_size is not None:
            if type(patch_size) is not list:
                self.patch_size = patch_size
            else:
                self.patch_size = patch_size[0]
        else:
            self.patch_size = 1024
        self.L_size = self.patch_size // self.sf

    def __getitem__(self, index):
        if self.phase == 'train':
            if self.get_image is False:
                img_H = self.paths_H[index]
                img_H = imread_uint(img_H)
                img_H = uint2single(img_H)
                img_H = modcrop(img_H, self.sf)
                if self.paths_L is not None:
                    img_L = self.paths_L[index]
                    img_L = imread_uint(img_L)
                    img_L = uint2single(img_L)
                else:
                    img_L = imresize_np(img_H, 1 / self.sf, True)
            else:
                img_H = self.paths_H[index]
                img_H = modcrop(img_H, self.sf)

                if self.paths_L is not None:
                    img_L = self.paths_L[index]
                else:
                    img_L = imresize_np(img_H, 1 / self.sf, True)
            H, W, C = img_L.shape
            if H < self.L_size or W < self.L_size:
                return None

            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            mode = random.randint(0, 7)
            img_L, img_H = augment_img(img_L, mode=mode), augment_img(img_H, mode=mode)
        else:
            img_H = self.paths_H[index]
            img_H = imread_uint(img_H)
            img_H = uint2single(img_H)
            img_H = modcrop(img_H, self.sf)
            if self.paths_L is not None:
                img_L = self.paths_L[index]
                img_L = imread_uint(img_L)
                img_L = uint2single(img_L)
            else:
                img_L = imresize_np(img_H, 1 / self.sf, True)
        img_H, img_L = single2tensor3(img_H), single2tensor3(img_L)
        return {'K': [img_L], 'H': [img_H], 'path': str(self.paths_H[index])}

    def __len__(self):
        return len(self.paths_H)

    def load_dataset(self, task_idx):
        if self.phase == 'train':
            if 'nyu' in self.opt['dataroot_H'][task_idx] or 'IXI' in self.opt['dataroot_H'][task_idx]:
                self.get_image = False
            else:
                self.get_image = True
        else:
            self.get_image = False

        self.get_image = False
        print('get_image', self.get_image)
        self.paths_H = get_image_paths(self.opt['dataroot_H'][task_idx], get_image=self.get_image)
        self.paths_L = get_image_paths(self.opt['dataroot_L'][task_idx], get_image=self.get_image)
        print('task {} {} load {} image'.format(task_idx, self.phase, len(self.paths_H)))
        patch_size = self.opt.get('H_size')
        if patch_size is not None:
            if type(patch_size) is not list:
                self.patch_size = patch_size
            else:
                self.patch_size = patch_size[task_idx]
        else:
            self.patch_size = 1024
        self.L_size = self.patch_size // self.sf
        return