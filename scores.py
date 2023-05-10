import copy
import math
from functools import partial
from inspect import isfunction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torch import einsum, nn, optim
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm

from dataset import create_loader, select_dataset
from model import select_model
from utils.util import set_seed

import argparse
import os
import os.path as osp
import shutil

import numpy as np
import torch as th
import torch.distributed as dist

from model import select_model
from model.set_diffusion import dist_util, logger
from model.set_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from utils.path import set_folder
from utils.util import count_params, set_seed

DIR = set_folder()

from fid import calculate_fid_given_paths
import lpips

def fid(real, fake, gpu, batch_size=50, dims=2048):
    print('Calculating FID...')
    print('real dir: {}'.format(real))
    print('fake dir: {}'.format(fake))
    # command = 'python -m pytorch_fid {} {} --gpu {}'.format(real, fake, gpu)  # pytorch-fid 0.1.1
    #command = 'python -m pytorch_fid {} {} --device cuda:{}'.format(real, fake, gpu)  # pytorch-fid 0.2.1
    # command = 'python -m pytorch_fid {} {}'.format(real, fake)
    #os.system(command)

    device = torch.device(gpu)
    fid_score = calculate_fid_given_paths((real, fake), batch_size=batch_size, device=device, dims=dims)
    return fid_score


def LPIPS(root):
    print('Calculating LPIPS...')
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    model = loss_fn_vgg
    model.cuda()

    files = os.listdir(root)
    data = {}
    for file in tqdm(files, desc='loading data'):
        cls = file.split('_')[0]
        idx = int(file.split('_')[1][:-4])
        img = lpips.im2tensor(cv2.resize(lpips.load_image(os.path.join(root, file)), (32, 32)))
        data.setdefault(cls, {})[idx] = img

    classes = set([file.split('_')[0] for file in files])
    res = []
    for cls in tqdm(classes):
        temp = []
        files_cls = [file for file in files if file.startswith(cls + '_')]
        for i in range(0, len(files_cls) - 1, 1):
            # print(i, end='\r')
            for j in range(i + 1, len(files_cls), 1):
                img1 = data[cls][i].cuda()
                img2 = data[cls][j].cuda()

                d = model(img1, img2, normalize=True)
                temp.append(d.detach().cpu().numpy())
        res.append(np.mean(temp))
    lpips_score = np.mean(res)
    print(lpips_score)
    return lpips_score


def generate_from_batch(args, model, batch, n_samples):
    batch_size = batch.size(0)
    if args.model == "ddpm":
        c = None
    else:
        c = model.sample_conditional(batch, n_samples)["c"]

        c = c.unsqueeze(1) # attention here
        c = torch.repeat_interleave(c, args.k * n_samples, dim=1)
        c = c.view(-1, c.size(-2), c.size(-1))
        print(c.size())
    sample_fn = (
        model.diffusion.p_sample_loop
        if not args.use_ddim
        else model.diffusion.ddim_sample_loop
    )
   
    sample = sample_fn(
        model.generative_model,
        (
            batch_size * n_samples * args.k,
            args.in_channels,
            args.image_size,
            args.image_size,
        ),
        c=c,
        clip_denoised=args.clip_denoised,
    )
    return sample

def to_images(tensors):
    images = tensors.cpu().detach().numpy()
    images = ((images + 1) / 2)
    images[images < 0] = 0
    images[images > 1] = 1
    images = images * 255
    images_out = Image.fromarray(images.astype('uint8'))
    return images_out

def eval_scores(args, dataset, model, n_cond, real_dir, fake_dir, transform):
    #if os.path.exists(fake_dir):
    #    shutil.rmtree(fake_dir)
    os.makedirs(fake_dir, exist_ok=True)
    #if os.path.exists(real_dir):
    #    shutil.rmtree(real_dir)
    os.makedirs(real_dir, exist_ok=True)

    data = dataset.images.reshape(-1, -1, args.in_channels, args.image_size, args.image_size)
    per = np.random.permutation(data.shape[1])
    data = data[:, per, :, :, :]

    num = n_cond
    data_for_gen = data[:, :num, :, :, :]
    data_for_fid = data[:, num:num+128, :, :, :]
    if os.path.exists(real_dir):
        for cls in tqdm(range(data_for_fid.shape[0]), desc='preparing real images'):
            for i in range(128):
                if data_for_fid.shape[1] < 128:
                    idx = np.random.choice(data_for_fid.shape[1], 1).item()
                else:
                    idx = i
            #for i in range(data_for_fid.shape[1]):
                #idx=i
                imgpath = os.path.join(real_dir, '{}_{}.png'.format(cls, str(i).zfill(3)))
                if not os.path.exists(imgpath):
                    real_img = data_for_fid[cls, idx, :, :, :]
                    real_img *= 255
                    real_img = Image.fromarray(np.uint8(real_img))
                    real_img.save(imgpath, 'png')

    if os.path.exists(fake_dir):
        for cls in tqdm(range(data_for_gen.shape[0]), desc='generating fake images'):
            for i in range(128):
                imgpath = os.path.join(fake_dir, '{}_{}.png'.format(cls, str(i).zfill(3)))
                if not os.path.exists(imgpath):
                    #idx = np.random.choice(data_for_gen.shape[1], n_cond)
                    imgs = data_for_gen#[cls, idx, :, :, :]
                    imgs = torch.cat([transform(img).unsqueeze(0) for img in imgs], dim=0).unsqueeze(0).cuda()
                    fake_imgs = generate_from_batch(args, model, imgs, 1)
                    output = to_images(fake_imgs)
                    output.save(imgpath, 'png')

    fid_score=fid(real_dir, fake_dir, int(args.gpu))
    lpips_score=LPIPS(fake_dir)

    shutil.rmtree(real_dir)
    shutil.rmtree(fake_dir)
    return fid_score, lpips_score








def main():
    args = create_argparser().parse_args()
    print(args)

    model = select_model(args)(args)
    print(count_params(model))
    
    #print(args.model_path)
    _path = list(args.model_path.split("/"))
    for j in range(len(_path)):
        _p = list(_path[j].split("_"))
        _p = [ i for i in _p if i not in ["", None, "None", "none"] ]
        _path[j] = "_".join(_p)
    #_path = list(args.model_path.split("_"))
    #_path = [ i for i in _path if i not in ["", None, "None", "none"] ]
    model_path = "/".join(_path)
    #print(model_path)

    model.load_state_dict(
        dist_util.load_state_dict(osp.join(DIR, model_path), map_location="cpu")
    )
    model.to(args.device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    dataset = select_dataset(args, "test")

    n_exps = args.n_exps

    if os.path.exists(args.eval_ckpt):
        eval_ckpt = torch.load(args.eval_ckpt)
        fid_scores, lpips_scores = eval_ckpt['fid'], eval_ckpt['lpips']
        n_exps = args.n_exps - len(fid_scores)
    else:
        os.makedirs(os.path.dirname(args.eval_ckpt), exist_ok=True)
        fid_scores = []
        lpips_scores=[]
        n_exps = args.n_exps

    transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)

    fid_scores=[]
    lpips_scores=[]
    for i in range(n_exps):
        fid_score, lpips_score = eval_scores(args, dataset, model, args.n_cond, args.real_dir, args.fake_dir, transform)
        fid_scores.append(fid_score)
        lpips_scores.append(lpips_score)
        torch.save({'fid': fid_scores, 'lpips': lpips_scores}, args.eval_ckpt)
    fid_out = sum(fid_scores) / len(fid_scores)
    lpips_out = sum(lpips_scores) / len(lpips_scores)

    with open(args.eval_path, 'a') as writer:
        fid_scores_str = ", ".join(["%.2f" % (x,) for x in fid_scores])
        lpips_scores_str = ", ".join(["%.4f" % (x,) for x in lpips_scores])
        writer.write("%s:\tFID: %.2f (%s)\tLPIPS: %.4f (%s)\n" % (args.dataset+"_"+str(args.n_cond), fid_out, fid_scores_str, lpips_out, lpips_scores_str)) 




def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=32,
        batch_size_eval=32,
        use_ddim=False,
        model_path="",
        k=1,  # multiplier for conditional samples
        model="vfsddpm",
        dataset="cifar100",
        pool='cls', # mean, mean_patch
        image_size=32,
        sample_size=5,
        patch_size=8,
        hdim=256,
        in_channels=3,
        encoder_mode="vit",
        context_channels=256,
        num_classes=1,
        mode_context="deterministic",
        mode_conditional_sampling="out-distro",
        mode_conditioning='bias', # film, lag, 
        augment=False,
        device="cuda",
        data_dir="data",
        transfer=False,
        n_exps=3,
        n_cond=10,
        eval_ckpt=None
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_dir')
    parser.add_argument('--real_dir')
    parser.add_argument('--eval_path')
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    s = set_seed(0)
    main()
    