import argparse
from AdamTTUR import AdamTTUR
from networks import Discriminator, Generator
import json
import math
import metrics
from modules.losses.GradientPenalty import compute_grad_penalty
from modules.losses.PerceptualLoss import PerceptualLoss
import numpy as np
import os
from data import SRDataset
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torchvision import utils
from typing import Dict
import wandb

IMAGE_UPLOAD_INTERVAL = 500
LOSS_WEIGHT_KEYS = ['percep_weight', 'pixel_weight', 'advers_weight']
RESULTS_DIR = './results'


def train_and_eval(gen_net: dict, gen_params: dict, disc_net: dict, disc_params: dict, percep_params: dict,
                   train_data: str, valid_data: str, batch_size: int, loss_weights: Dict[str, float]) -> None:
    timestamp = time.time()
    torch.autograd.set_detect_anomaly(True)

    disc_grad_coeff = disc_params['grad_penalty']

    generator = Generator(**gen_net)
    discriminator = Discriminator(**disc_net)

    wandb.init(project='Single-Image Super-Resolution', name=str(timestamp),
               tags=['instance normalisation' if gen_net['instance_norm'] else 'batch normalisation'])
    wandb.config.update({'generator': gen_net, 'discriminator': disc_net, 'perceptual_loss': percep_params,
                         'batch_size': batch_size, 'train_data': train_data, 'valid_data': valid_data,
                         'generator_optim': gen_params, 'discriminator_optim': disc_params,
                         'generator_loss': loss_weights})
    wandb.watch(generator)
    wandb.watch(discriminator)

    disc_params.pop('grad_penalty')

    _device = torch.device('cpu')
    if torch.cuda.is_available():
        _device = torch.device('cuda')
        generator.to(device=_device)
        discriminator.to(device=_device)

    # ============================== Begin Training ============================== #

    training_dataset = SRDataset(root=train_data, load_device=_device)
    data_loader = DataLoader(training_dataset, batch_size, shuffle=True)

    gen_optimiser = AdamTTUR(generator.parameters(), **gen_params)
    disc_optimiser = AdamTTUR(discriminator.parameters(), **disc_params)

    perceptual_loss = PerceptualLoss(**percep_params)
    perceptual_loss.to(device=_device)

    generator.train()
    discriminator.train()
    data_iter = data_loader.__iter__()
    for train_step in range(0, 40000):
        # Train discriminator
        try:
            samples, targets = data_iter.__next__()
            super_resolved = generator.forward(samples)

            epsilons = torch.rand((samples.shape[0], 1, 1, 1), device=_device)
            interpolated = epsilons * targets + (1.0 - epsilons) * super_resolved
            interpolated.requires_grad_(True)

            fakes_prob = discriminator.forward(super_resolved)
            reals_prob = discriminator.forward(targets)

            gradient_penalty = compute_grad_penalty(discriminator, interpolated)

            disc_loss = torch.mean(fakes_prob - reals_prob + disc_grad_coeff * gradient_penalty)
            disc_loss.backward()

            disc_optimiser.step()
        except StopIteration:
            break
        except IndexError:
            break

        # Train generator
        try:
            samples, targets = data_iter.__next__()
            super_resolved = generator.forward(samples)

            gen_optimiser.zero_grad()

            gen_pixel_loss = F.l1_loss(super_resolved, targets)
            gen_pixel_loss = torch.mul(gen_pixel_loss, loss_weights['pixel_weight'])
            gen_pixel_loss.backward(retain_graph=True)

            fakes_prob = discriminator.forward(super_resolved)
            gen_adversarial_loss = (-1.0 * loss_weights['advers_weight']) * torch.mean(fakes_prob)
            gen_adversarial_loss.backward(retain_graph=True)

            # Bring images to to range [0, 1] for perceptual loss' VGG network
            super_resolved, targets = torch.add(super_resolved, 1.0), torch.add(targets, 1.0)
            super_resolved, targets = torch.div(super_resolved, 2.0), torch.div(targets, 2.0)
            gen_content_loss, gen_style_loss = perceptual_loss.forward(super_resolved, targets)
            gen_content_loss = torch.mul(gen_content_loss, loss_weights['percep_weight'])
            gen_content_loss.backward(retain_graph=True)
            gen_style_loss = torch.mul(gen_style_loss, loss_weights['percep_weight'])
            gen_style_loss.backward()

            gen_optimiser.step()
        except StopIteration:
            break
        except IndexError:
            break

        wandb.log({'generator_loss': {'perceptual': gen_content_loss + gen_style_loss, 'pixel': gen_pixel_loss,
                                      'adversarial': gen_adversarial_loss},
                   'discriminator_loss': disc_loss}, step=train_step)

        if train_step % IMAGE_UPLOAD_INTERVAL == 0:
            super_resolved = super_resolved.to(device='cpu')
            targets = targets.to(device='cpu')
            metrics.norm_inplace(super_resolved)
            metrics.norm_inplace(targets)
            _super_resolved_images = []
            _target_images = []
            for idx in range(super_resolved.shape[0]):
                _super_resolved_images.append(wandb.Image(to_pil_image(super_resolved[idx, :, :, :], mode='RGB')))
                _target_images.append(wandb.Image(to_pil_image(targets[idx, :, :, :], mode='RGB')))
            wandb.log({'super_resolved': _super_resolved_images, 'targets': _target_images})

    # ============================== Begin Validation ============================== #

    results_dir = os.path.join(RESULTS_DIR, str(timestamp))
    images_dir = os.path.join(results_dir, valid_data)
    os.makedirs(images_dir, exist_ok=True)

    torch.save(generator.state_dict(), os.path.join(results_dir, 'generator_{:s}.pth'.format(str(timestamp))))
    torch.save(discriminator.state_dict(), os.path.join(results_dir, 'discriminator_{:s}.pth'.format(str(timestamp))))

    del discriminator
    del perceptual_loss
    torch.cuda.empty_cache()

    generator.eval()

    validation_dataset = SRDataset(root=valid_data, load_device=_device, sort=True)
    data_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    frechet_inception = metrics.FrechetInception(_device)

    psnrs = []
    ssims = []
    for image_idx, (sample_image, target_image) in enumerate(data_loader):
        print(sample_image.dtype)
        print(type(sample_image))
        with torch.no_grad():
            super_resolved = generator.forward(sample_image)

        metrics.norm_inplace(super_resolved)
        metrics.norm_inplace(target_image)

        _psnr = metrics.peak_signal_to_noise(super_resolved, target_image)
        _ssim = metrics.struct_similarity(super_resolved, target_image)
        frechet_inception.add_codes(super_resolved, target_image)

        psnrs.append(_psnr.item())
        ssims.append(_ssim.item())

        utils.save_image(torch.squeeze(super_resolved),
                         os.path.join(images_dir, os.path.split(validation_dataset.image_paths[image_idx][0])[-1]))

    data = [(idx, _psnr) for idx, _psnr in enumerate(psnrs)]
    table = wandb.Table(data=data, columns=['Image Index', 'PSNR'])
    wandb.log({'psnr_plot': wandb.plot.scatter(table, 'Image Index', 'PSNR')})

    data = [(idx, _ssim) for idx, _ssim in enumerate(ssims)]
    table = wandb.Table(data=data, columns=['Image Index', 'SSIM'])
    wandb.log({'ssim_plot': wandb.plot.scatter(table, 'Image Index', 'SSIM')})

    fid = frechet_inception.compute_distance()
    wandb.log({'frechet_inception_distance': fid,
               'ssim': {'mean': np.mean(ssims), 'min': np.min(ssims), 'max': np.max(ssims), 'std_dev': np.std(ssims)},
               'psnr': {'mean': np.mean(psnrs), 'min': np.min(psnrs), 'max': np.max(psnrs), 'std_dev': np.std(psnrs)}
               })


if __name__ == '__main__':
    gen_net = {'num_features': 64, 'num_res_blocks': 8, 'num_dense_layers': 16}
    disc_net = {'input_res': 256, 'num_features_start': 16, 'num_features_stop': 256, 'num_stacked_layers': 1,
                'lrelu_slope': 0.15}
    gen_params = {'percep_weight': 0.50, 'pixel_weight': 0.05, 'advers_weight': 0.45, 'lr': 0.0001, 'tau': 0.1,
                  'alpha': 0.5, 'memory': 5000.0}
    disc_params = {'grad_penalty': 10.0, 'lr': 0.0002, 'tau': 0.2, 'alpha': 0.3, 'memory': 500.0}
    percep_params = {'content_weight': 0.98, 'content_layer': 'relu2_1', 'style_layer': 'relu3_1'}

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default='MSCOCO', help='Training data folder in ./datasets')
    parser.add_argument('--valid-data', type=str, default='DIV2K', help='Validation data folder in ./datasets')
    parser.add_argument('--batch-norm', action='store_true', help='Use batch normalisation rather than instance '
                                                                  'normalisation in generator')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size to use during training')
    parser.add_argument('--gen-net', type=str, default=(str(gen_net).replace('\'', '"')),
                        help='JSON string representing parameters of Generator __init__ function')
    parser.add_argument('--disc-net', type=str, default=(str(disc_net).replace('\'', '"')),
                        help='JSON string representing parameters of Discriminator __init__ function')
    parser.add_argument('--percep-params', type=str, default=(str(percep_params)).replace('\'', '"'),
                        help='JSON string representing content weight between 0 and 1, and content and style layers in '
                             'VGG for perceptual loss. Form: {"content_weight":float, "content_layer":"convi_j", '
                             '"style_layer":"convi_j"} where i={1,2,3,4,5} and j={1,2,3,4}')
    parser.add_argument('--gen-params', type=str, default=(str(gen_params)).replace('\'', '"'),
                        help='Miscellaneous generator hyper-parameters.')
    parser.add_argument('--disc-params', type=str, default=(str(disc_params)).replace('\'', '"'),
                        help='Discriminator\'s Adam with TTUR optimiser parameters.')

    args = parser.parse_args()

    gen_net = json.loads(args.gen_net)
    gen_net['instance_norm'] = not args.batch_norm
    gen_params = json.loads(args.gen_params)
    disc_net = json.loads(args.disc_net)
    percep_params = json.loads(args.percep_params)
    loss_weights = {}
    loss_weights_sum = 0.0
    for key in LOSS_WEIGHT_KEYS:
        loss_weights[key] = float(gen_params[key])
        loss_weights_sum += loss_weights[key]
        gen_params.pop(key)
    disc_params = json.loads(args.disc_params)

    assert math.isclose(loss_weights_sum, 1.0)

    train_and_eval(gen_net, gen_params, disc_net, disc_params, percep_params, args.train_data, args.valid_data,
                   args.batch_size, loss_weights)
