import torch.nn as nn
from torch import autograd as autograd

from utils import *

class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer: Union[int, list] = [2, 7, 16, 25, 34], use_input_norm=True,
                 use_range_norm=False):
        super(VGGFeatureExtractor, self).__init__()

        model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        self.use_range_norm = use_range_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.list_outputs = isinstance(feature_layer, list)
        if self.list_outputs:
            self.features = nn.Sequential()
            feature_layer = [-1] + feature_layer
            for i in range(len(feature_layer) - 1):
                self.features.add_module('child' + str(i), nn.Sequential(
                    *list(model.features.children())[(feature_layer[i] + 1):(feature_layer[i + 1] + 1)]))
        else:
            self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])

        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_range_norm:
            x = (x + 1.0) / 2.0
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        if self.list_outputs:
            output = []
            for child_model in self.features.children():
                x = child_model(x)
                output.append(x.clone())
            return output
        else:
            return self.features(x)


class PerceptualLoss(nn.Module):

    def __init__(self, opt=None, feature_layer=[2, 7, 16, 25, 34], weights=[0.1, 0.1, 1.0, 1.0, 1.0], lossfn_type='l1',
                 reduction='mean',
                 use_input_norm=True, use_range_norm=False):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatureExtractor(feature_layer=feature_layer, use_input_norm=use_input_norm,
                                       use_range_norm=use_range_norm)
        self.lossfn_type = lossfn_type
        self.weights = weights
        if self.lossfn_type == 'l1':
            self.lossfn = nn.L1Loss(reduction=reduction)
        else:
            self.lossfn = nn.MSELoss(reduction=reduction)
        print(f'feature_layer: {feature_layer}  with weights: {weights}')

    def forward(self, x, gt):

        x_vgg, gt_vgg = self.vgg(x), self.vgg(gt.detach())
        loss = 0.0
        if isinstance(x_vgg, list):
            n = len(x_vgg)
            for i in range(n):
                loss += self.weights[i] * self.lossfn(x_vgg[i], gt_vgg[i])
        else:
            loss += self.lossfn(x_vgg, gt_vgg.detach())
        return loss

class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, reduction='mean'):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss(reduction=reduction)
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            def wgan_loss(input, target):
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        elif self.gan_type == 'softplusgan':
            def softplusgan_loss(input, target):
                return F.softplus(-input).mean() if target else F.softplus(input).mean()

            self.loss = softplusgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type in ['wgan', 'softplusgan']:
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):

        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss


def r1_penalty(real_pred, real_img):
    grad_real = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (
            path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty


class ContractLoss(nn.Module):
    def __init__(self, opt=None):
        super(ContractLoss, self).__init__()

    def forward(self, inp: list, gt):
        l = len(inp)
        device = gt.device
        s = [i.to(device) for i in inp]
        return sum(s) / l


class GradLoss(nn.Module):
    def __init__(self, opt=None):
        super(GradLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pre, gt):
        if pre.shape[1] != 1:
            grad_pre = get_gradient(pre)
        else:
            grad_pre = pre
        gt_g = transforms.Grayscale()(gt)
        grad_gt = get_gradient(gt_g)
        gradloss = self.l1(grad_pre, grad_gt)
        return gradloss


class FreqLoss(nn.Module):
    def __init__(self, opt=None):
        super(FreqLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pre, gt):
        freq_gt = torch.fft.fft2(gt, norm='ortho')
        freq_pre = torch.fft.fft2(pre, norm='ortho')
        tmp1 = self.l1(freq_pre.real, freq_gt.real)
        tmp2 = self.l1(freq_pre.imag, freq_gt.imag)
        return tmp1 + tmp2


class DiceLoss(nn.Module):
    def __init__(self, bin_wide, density):
        super(DiceLoss, self).__init__()
        self.bin_wide = bin_wide
        self.density = density

    def soft_dice_coeff(self, y_pred, y_true, class_i=None):
        smooth = 0.0001

        i = torch.sum(y_true, dim=(1, 2))
        j = torch.sum(y_pred, dim=(1, 2))
        intersection = torch.sum(y_true * y_pred, dim=(1, 2))

        score = (2. * intersection + smooth) / (i + j + smooth)

        if self.bin_wide:
            weight = self.density_weight(self.bin_wide[class_i], i, self.density[class_i])
            return (1 - score) * weight
        else:
            return (1 - score)

    def soft_dice_loss(self, y_pred, y_true, class_i=None):
        loss = self.soft_dice_coeff(y_true, y_pred, class_i)
        return loss.mean()

    def density_weight(self, bin_wide, gt_cnt, density):

        index = gt_cnt // bin_wide

        selected_density = [density[index[i].long()] for i in range(gt_cnt.shape[0])]
        selected_density = torch.tensor(selected_density).cuda()
        log_inv_density = torch.log(1 / (selected_density + 0.0001))

        return log_inv_density

    def __call__(self, y_pred, y_true, class_i=None):

        b = self.soft_dice_loss(y_true, y_pred, class_i)
        return b


if __name__ == '__main__':
    x = [
        torch.tensor([[[[1, 2, 3], [4, 5, 6]]]], dtype=torch.float32),
        torch.tensor([[[[2, 3, 4], [4, 5, 6]]]], dtype=torch.float32)
    ]
    y = torch.tensor([[[[6, 5, 4], [3, 2, 1]]]], dtype=torch.float32)
    print(x)
    print(y)
