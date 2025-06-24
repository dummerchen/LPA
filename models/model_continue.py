import transformers
from torch.optim import Adam
from torch.optim import lr_scheduler
from torchvision import transforms
from models.loss import *
from models.model_base import ModelBase
from models.network_promptnet import PromptNet
from models.network_msr import Msr
from models.select_network import define_G
from utils.utils_image import mkdir
from utils.utils_regularizers import regularizer_orth, regularizer_clip
from transformers import CLIPModel

class ModelContinue(ModelBase):
    """Train with pixel loss"""

    def __init__(self, opt, **kwargs):
        super(ModelContinue, self).__init__(opt)
        self.opt_train = self.opt['train']
        self.logger = get_logger(kwargs.get('phase'))
        self.prompt = define_G(opt['netG']['prompt'])
        self.prompt.freeze_extract = opt['netG']['freeze_extract']['net_type']
        self.backbone = define_G(opt['netG']['backbone'])
        self.netG = PromptNet(backbone=self.backbone, prompt=self.prompt)
        self.netG = self.model_to_device(self.netG, phase=kwargs.get('phase'))
        self.G_lossfn = []
        self.resume_weight = False
        self.prune = self.opt['netG']['freeze_extract']['prune']
        self.emb_dim = self.opt['netG']['prompt']['emb_dim'][0]
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(self.opt).to(self.device).eval()
        self.define_freeze_extract()
    def define_freeze_extract(self):
        self.freeze_extract = []
        if not isinstance(self.opt['netG']['freeze_extract']['net_type'], list):
            self.opt['netG']['freeze_extract']['net_type'] = [self.opt['netG']['freeze_extract']['net_type']]
        for type in self.opt['netG']['freeze_extract']['net_type']:
            if type in ['clip']:
                freeze_extract = CLIPModel.from_pretrained(
                    os.path.expanduser("~/.cache/huggingface/hub/clip-vit-base-patch32"), local_files_only=True)
                self.freeze_processor = transformers.CLIPImageProcessor.from_pretrained(
                    os.path.expanduser("~/.cache/huggingface/hub/clip-vit-base-patch32"), local_files_only=True)
                if self.prune:
                    print('Pruning CLIP')
                    freeze_extract.visual_projection = self.prune_network(freeze_extract.visual_projection, self.emb_dim)
            elif type == 'msr':
                freeze_extract = Msr()
                freeze_extract.load_state_dict(torch.load('./model_zoo/DDG_Encoder.pth'), strict=True)
                if self.prune:
                    print('Pruning MSR')
                    freeze_extract.conv1x1_3 = self.prune_network(freeze_extract.conv1x1_3, self.emb_dim)
            elif type == 'googlenet':
                freeze_extract = torchvision.models.googlenet(pretrained=True)
            elif type == 'vgg':
                freeze_extract = torchvision.models.vgg16(pretrained=True)
            else:
                freeze_extract = nn.Identity()
            self.freeze_extract.append(freeze_extract)

    def prune_network(self, old_image_proj, emb_dim, next_conv=None):
        if isinstance(old_image_proj, nn.Linear):
            new_image_proj = nn.Linear(old_image_proj.in_features, emb_dim, bias=old_image_proj.bias is not None)
            old_weights = old_image_proj.weight.data.cpu().numpy()
            new_weights = np.zeros(shape=(emb_dim, old_image_proj.in_features), dtype=np.float32)
            idx0 = np.argsort(abs(old_weights), axis=0)
            # 512 768 - > 180 768
            for i in range(old_image_proj.in_features):
                new_weights[:, i] = np.delete(old_weights[:, i], idx0[:old_image_proj.out_features - emb_dim, i])
                if old_image_proj.bias is not None:
                    pass
            device = old_image_proj.weight.device
            new_image_proj.weight.data = torch.from_numpy(new_weights)
            if device != 'cpu':
                new_image_proj = new_image_proj.to(device)
            if next_conv is not None:
                pass
            if old_image_proj.bias is not None:
                bias_numpy = old_image_proj.bias.data.cpu().numpy()
                bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
                new_image_proj.bias.data = torch.from_numpy(bias)
        elif isinstance(old_image_proj, nn.Conv2d):
            importance = old_image_proj.weight.abs().sum(dim=(1, 2, 3))  # 对每个输出通道的权重求绝对值和

            # 获取按权重重要性排序的索引，并保留前180个
            _, indices = torch.topk(importance, emb_dim)

            # 构建裁剪后的新卷积层A_pruned
            new_image_proj = nn.Conv2d(in_channels=old_image_proj.in_channels, out_channels=emb_dim, kernel_size=1, bias=True if old_image_proj.bias is not None else False)

            # 将保留的权重和偏置复制到A_pruned中
            new_image_proj.weight.data = old_image_proj.weight.data[indices]
            new_image_proj.bias.data = old_image_proj.bias.data[indices]
        return new_image_proj

    def init_train(self):
        self.load()
        self.netG.train()
        self.define_loss()
        self.define_optimizer()
        self.load_optimizers()
        self.define_scheduler()
        self.log_dict = OrderedDict()

    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            self.logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')

    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            self.logger.info('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)
            self.resume_weight = True

    def save(self, iter_label, save_best=False):
        temp_save_dir = self.save_dir
        if save_best is True:
            self.save_dir = os.path.join(self.save_dir, 'best')
            mkdir(self.save_dir)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        self.save_dir = temp_save_dir


    def define_loss(self):
        self.G_lossfn_type = self.opt_train['G_lossfn_type']
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

        def get_loss_fn(lossfn_type: str):
            if lossfn_type == 'l2sum':
                return nn.MSELoss(reduction='sum').to(self.device)
            else:
                try:
                    return (eval(lossfn_type)(self.opt).to(self.device))
                except Exception as e:
                    raise NotImplementedError('Loss type [{:s}] is not found and {}'.format(lossfn_type, e))

        for lossfn_type in self.G_lossfn_type:
            if type(lossfn_type) is list:
                ts = []
                for lt in lossfn_type:
                    ts.append(get_loss_fn(lt))
                self.G_lossfn.append(ts)
            else:
                self.G_lossfn.append(get_loss_fn(lossfn_type))


    def define_optimizer(self, print_optimize=False, task_id=0):
        G_optim_params = []
        optimize_params = []
        trainable_params = 0
        all_param = 0
        freeze_layers = self.opt['freeze_layers'] if self.opt['freeze_layers'] is not None else []
        unfreeze_layers = self.opt['unfreeze_layers'] if self.opt['unfreeze_layers'] is not None else []
        for k, v in self.netG.named_parameters():
            all_param += v.numel()
            if 'all' in freeze_layers:
                v.requires_grad = False
                continue
            for pattern in freeze_layers:
                if re.search(pattern, k):
                    v.requires_grad = False

        for k, v in self.netG.named_parameters():
            for pattern in unfreeze_layers:
                if re.search(pattern, k):
                    v.requires_grad = True
            if v.requires_grad:
                G_optim_params.append(v)
                trainable_params += v.numel()
                optimize_params.append(k)
                print('Params [{:s}] will do optimize !'.format(k))

        self.logger.info(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'][task_id],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'])
        else:
            raise NotImplementedError


    def define_scheduler(self, current_step=-1, task_id=0):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers = lr_scheduler.MultiStepLR(optimizer=self.G_optimizer,
                                                       milestones=self.opt_train['G_scheduler_milestones'][task_id][
                                                                  :-1],
                                                       gamma=self.opt_train['G_scheduler_gamma'],
                                                       last_epoch=current_step
                                                       )
        elif self.opt_train['G_scheduler_type'] == 'StepLR':
            self.schedulers = lr_scheduler.StepLR(optimizer=self.G_optimizer,
                                                  step_size=self.opt_train['G_scheduler_milestones'][task_id][:-1],
                                                  gamma=self.opt_train['G_scheduler_gamma'],
                                                  )
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers = lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer,
                                                                       self.opt_train['G_scheduler_periods'],
                                                                       self.opt_train[
                                                                           'G_scheduler_restart_weights'],
                                                                       self.opt_train['G_scheduler_eta_min'],
                                                                       )
        else:
            raise NotImplementedError

    def resume(self, current_step=-1, task_id=0):
        print('resume ', task_id)
        self.define_scheduler(current_step=current_step, task_id=task_id)

        flag = self.opt['task'][-4:] if self.opt['flag'] is None else self.opt['flag']

        if type(self.netG).__name__ in ['DataParallel', 'DistributedDataParallel']:
            self.netG.module.prompt.process_task_add(task_id, flag=flag)
        else:
            self.netG.prompt.process_task_add(task_id, flag=flag)

    def extract_features(self, x, device='cpu', return_tensors="pt", do_convert_rgb=False):
        '''
            x : 0-1
        '''

        ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        vit_transforms = transforms.Compose([

            transforms.Resize(size=248),
            transforms.CenterCrop(size=(224, 224)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        outs = []
        for i, freeze_extract in enumerate(self.freeze_extract):
            freeze_extract = freeze_extract.to(device).eval()
            with torch.no_grad():
                if self.opt['netG']['freeze_extract']['net_type'][i] == 'clip':
                    inputs = self.freeze_processor(images=(x * 255).long(), return_tensors=return_tensors, do_rescale=True,
                                                   do_convert_rgb=do_convert_rgb).to(
                        device)
                    inputs.data['pixel_values'] = inputs.data['pixel_values'].type(torch.float32)
                    img_emb = freeze_extract.get_image_features(**inputs)
                    out = img_emb
                elif self.opt['netG']['freeze_extract']['net_type'][i] == 'msr':
                    x = transforms.CenterCrop(size=(96, 96))(x)
                    out = freeze_extract(x)
                elif self.opt['netG']['freeze_extract']['net_type'][i] in ['googlenet', 'vgg']:
                    x = transforms.CenterCrop(size=(96, 96))(x)
                    out = freeze_extract(x)

            outs.append(out)
        return outs

    def feed_data(self, data, need_H=True):
        self.L = []
        for d in data['L']:
            if type(d) is list:
                self.L.append([dd.float().to(self.device, non_blocking=True) for dd in d])
            else:
                self.L.append(d.float().to(self.device, non_blocking=True))

        if need_H:
            self.H = []
            for d in data['H']:
                if type(d) is list:
                    self.H.append([dd.float().to(self.device, non_blocking=True) for dd in d])
                else:
                    self.H.append(d.float().to(self.device, non_blocking=True))

    def netG_forward(self, current_step=None):
        self.E = self.netG(self.L, func=self.extract_features)
        if type(self.E) is list:
            if type(self.H) is list:
                # 自动用HR填充不足
                if len(self.H) < len(self.E):
                    for _ in range(len(self.E) - len(self.H)):
                        self.H.insert(-1, self.H[0])
                else:
                    self.H = self.H[:len(self.E)]
                if len(self.L) < len(self.E):
                    self.L = self.L + self.L * (len(self.E) - len(self.L))
                else:
                    self.L = self.L[:len(self.E)]
            else:
                self.H = [self.H] * len(self.E)
        else:
            self.E = [self.E]
            self.H = [self.H]

    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        G_loss = []
        for i, func in enumerate(self.G_lossfn):
            if type(func) is list:
                for j, f in enumerate(func):
                    G_loss.append(self.G_lossfn_weight[i][j] * f(self.E[i], self.H[i]))
                    self.log_dict['G' + self.G_lossfn_type[i][j]] = G_loss[-1]
            else:
                G_loss.append(self.G_lossfn_weight[i] * func(self.E[i], self.H[i]))
                self.log_dict['G' + self.G_lossfn_type[i]] = G_loss[-1]
        if hasattr(self.netG.module.prompt, 'loss'):
            self.log_dict['G prompt loss'] = self.netG.module.prompt.loss
            G_loss = sum(G_loss) + self.netG.module.prompt.loss
            G_loss.backward()
            self.netG.module.prompt.loss = 0
        else:
            G_loss = sum(G_loss)
            G_loss.backward()
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'],
                                           norm_type=2)
        self.G_optimizer.step()

        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train[
            'G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train[
            'G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        self.log_dict['G_loss'] = G_loss.item()
        self.log_dict['epoch'] = current_step
        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    def ttest(self, img_lq: list, scale: int, tile=None, task_id=None):
        if tile is None:
            output = self.netG(img_lq, self.extract_features, task_id=task_id)

        else:
            b1, c1, h1, w1 = img_lq[0].size()
            tile = min(tile, h1, w1)
            tile_overlap = 8
            sf = scale

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h1 - tile, stride)) + [h1 - tile]
            w_idx_list = list(range(0, w1 - tile, stride)) + [w1 - tile]
            E = []
            W = []

            flag = False
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patchs = []
                    for img_ in img_lq:
                        b, c, h, w = img_.shape
                        s = int(h / h1)
                        in_patchs.append(
                            img_[:, :, h_idx * s:(h_idx + tile) * s, w_idx * s:(w_idx + tile) * s])
                    out_patchs = self.netG(in_patchs, self.extract_features, task_id=task_id)
                    for i, out_patch in enumerate(out_patchs):
                        if type(out_patch) is list:
                            if flag is False:
                                E.append([])
                                W.append([])

                            for j, out in enumerate(out_patch):
                                b, c, h, w = out.shape
                                if flag is False:
                                    if h != tile * sf:
                                        E[i].append(torch.zeros(b, c, h1, w1).type_as(out))
                                        W[i].append(torch.zeros(b, c, h1, w1).type_as(out))
                                    else:
                                        E[i].append(torch.zeros(b, c, h1 * sf, w1 * sf).type_as(out))
                                        W[i].append(torch.zeros(b, c, h1 * sf, w1 * sf).type_as(out))
                                if h != tile * sf:
                                    out_mask = torch.ones_like(out)
                                    E[i][j][:, :, h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out)
                                    W[i][j][:, :, h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_mask)
                                else:
                                    out_mask = torch.ones_like(out)
                                    E[i][j][:, :, h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(
                                        out)
                                    W[i][j][:, :, h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(
                                        out_mask)

                        else:
                            if flag is False:
                                E.append(torch.zeros(b1, c1, h1 * sf, w1 * sf).type_as(out_patch))
                                W.append(torch.zeros(b1, c1, h1 * sf, w1 * sf).type_as(out_patch))

                            out_patch_mask = torch.ones_like(out_patch)
                            E[i][:, :, h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
                            W[i][:, :, h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(
                                out_patch_mask)
                    flag = True
            output = []
            for e, w in zip(E, W):
                if type(e) is list:
                    output.append([])
                    for i, j in zip(e, w):
                        output[-1].append(i.div(j))
                else:
                    output.append(e.div(w))
        return output

    def current_log(self):
        return self.log_dict
