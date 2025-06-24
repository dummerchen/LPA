import os

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from utils import get_logger
from utils.utils_bnorm import merge_bn, tidy_sequential


class ModelBase():
    def __init__(self, opt):
        self.opt = opt
        self.save_dir = opt['path']['models']
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        if self.is_train:
            self.logger = get_logger('train')
        else:
            self.logger = get_logger('test')


    def init_train(self):
        pass

    def load(self):
        pass

    def save(self, label):
        pass

    def define_loss(self):
        pass

    def define_optimizer(self):
        pass

    def define_scheduler(self, task_id=None):
        pass



    def feed_data(self, data):
        pass

    def optimize_parameters(self, current_step):
        pass

    def current_visuals(self):
        pass

    def current_losses(self):
        pass

    def update_learning_rate(self, n):
        self.schedulers.step(n)

    def current_learning_rate(self):
        return self.schedulers.get_lr()[0]

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag


    def print_network(self):
        pass

    def info_network(self):
        pass

    def print_params(self):
        pass

    def info_params(self):
        pass

    def get_bare_model(self, network):
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def model_to_device(self, network, **kwargs):

        if kwargs.get('phase') == 'train':
            network = network.to(self.device)
            if self.opt['dist']:
                find_unused_parameters = self.opt.get('find_unused_parameters', True)
                use_static_graph = self.opt.get('use_static_graph', False)
                network = DistributedDataParallel(network, device_ids=[torch.cuda.current_device()],
                                                  find_unused_parameters=find_unused_parameters)
                if use_static_graph:
                    print('Using static graph. Make sure that "unused parameters" will not change during training loop.')
                    network._set_static_graph()
            else:
                network = DataParallel(network)
        return network

    def describe_network(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += 'Networks name: {}'.format(network.__class__.__name__) + '\n'
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), network.parameters()))) + '\n'
        msg += 'Net structure:\n{}'.format(str(network)) + '\n'
        return msg

    def describe_params(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape',
                                                                        'param_name') + '\n'
        for name, param in network.state_dict().items():
            if not 'num_batches_tracked' in name:
                v = param.data.clone().float()
                msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(),
                                                                                          v.std(), v.shape, name) + '\n'
        return msg


    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        network = self.get_bare_model(network)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            if 'extract' or 'processor' in key:
                continue
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True, param_key='params'):
        network = self.get_bare_model(network)
        if strict:
            state_dict = torch.load(load_path)
            if param_key in state_dict.keys():
                state_dict = state_dict[param_key]
            network.load_state_dict(state_dict, strict=strict)
        else:
            pretrain_weights = torch.load(load_path)
            if param_key in pretrain_weights.keys():
                pretrain_weights = pretrain_weights[param_key]
            scratch_dict = network.state_dict()
            target_dict = {}
            for k in scratch_dict.keys():
                if k in pretrain_weights.keys() and scratch_dict[k].shape == pretrain_weights[k].shape:
                    target_dict[k] = pretrain_weights[k].clone()
                elif k.replace('backbone.', '') in pretrain_weights.keys() and scratch_dict[k].shape == pretrain_weights[k.replace('backbone.','')].shape:
                    target_dict[k] = pretrain_weights[k.replace('backbone.', '')].clone()
                else:
                    target_dict[k] = scratch_dict[k].clone()
            network.load_state_dict(target_dict, strict=True)
            del pretrain_weights, scratch_dict

    def save_optimizer(self, save_dir, optimizer, optimizer_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, optimizer_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def load_optimizer(self, load_path, optimizer):
        optimizer.load_state_dict(
            torch.load(load_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))

    def update_E(self, current_step=1):
        decay = min(self.opt_train['E_warmup'] + (self.opt_train['E_decay'] - self.opt_train['E_warmup']) // \
                    self.opt_train['E_warmup_iters'] * current_step, self.opt_train['E_decay'])
        netG = self.get_bare_model(self.netG)
        netG_params = dict(netG.named_parameters())
        netE_params = dict(self.netE.named_parameters())
        for k in netG_params.keys():
            netE_params[k].data.mul_(decay).add_(netG_params[k].data, alpha=1 - decay)


    def merge_bnorm_train(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
        self.define_optimizer()
        self.define_scheduler()

    def merge_bnorm_test(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
