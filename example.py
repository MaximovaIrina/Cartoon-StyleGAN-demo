import os
import argparse
from tqdm import tqdm

import cv2
from PIL import Image
import PIL.Image as pilimg

import torch
import numpy as np
from torchvision.utils import make_grid

from utils import download_pretrained_model
from model import Generator

device = 'cpu'

def init_generators():
    download_pretrained_model(download_all=True)

    # Genearaotr1
    network1 = torch.load('networks/ffhq256.pt', map_location=device)
    g1 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    g1.load_state_dict(network1["g_ema"], strict=False)
    trunc1 = g1.mean_latent(4096)

    # Generator2
    network2 = torch.load('networks/NaverWebtoon_StructureLoss.pt', map_location=device)
    g2 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    g2.load_state_dict(network2["g_ema"], strict=False)
    trunc2 = g2.mean_latent(4096)

    # Generator3
    network3 = torch.load('networks/Disney_StructureLoss.pt', map_location=device)
    g3 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    g3.load_state_dict(network3["g_ema"], strict=False)
    trunc3 = g3.mean_latent(4096)

    # Generator4
    network4 = torch.load('networks/Metface_StructureLoss.pt', map_location=device)
    g4 = Generator(256, 512, 8, channel_multiplier=2).to(device)
    g4.load_state_dict(network4["g_ema"], strict=False)
    trunc4 = g4.mean_latent(4096)

    return ((g1, trunc1), (g2, trunc2), (g3, trunc3), (g4, trunc4))


def generate_images(generators, number_of_img, number_of_step, outdir, tqdm_disable=False):
    g_trunc1, g_trunc2, g_trunc3, g_trunc4 = generators
    g1, trunc1 = g_trunc1
    g2, trunc2 = g_trunc2
    g3, trunc3 = g_trunc3
    g4, trunc4 = g_trunc4
    swap = True
    swap_layer_num = 2

    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    with torch.no_grad():
        latent1 = torch.randn(1, 14, 512, device=device)
        latent1 = g1.get_latent(latent1)
        latent_interp = torch.zeros(1, latent1.shape[1], latent1.shape[2]).to(device)
        for i in range(number_of_img):
            latent2 = torch.randn(1, 14, 512, device=device)
            latent2 = g1.get_latent(latent2)
            for j in tqdm(range(number_of_step), desc=F"Image {i}/{number_of_img}, step:", disable=tqdm_disable):
                latent_interp = latent1 + (latent2-latent1) * float(j/(number_of_step-1))
                imgs_gen1, save_swap_layer = g1([latent_interp],
                                                input_is_latent=True,
                                                truncation=0.7,
                                                truncation_latent=trunc1,
                                                swap=swap, swap_layer_num=swap_layer_num,
                                                randomize_noise=False)
                imgs_gen2, _ = g2([latent_interp],
                                  input_is_latent=True,
                                  truncation=0.7,
                                  truncation_latent=trunc2,
                                  swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
                                  )
                imgs_gen3, _ = g3([latent_interp],
                                  input_is_latent=True,
                                  truncation=0.7,
                                  truncation_latent=trunc3,
                                  swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
                                  )
                imgs_gen4, _ = g4([latent_interp],
                                  input_is_latent=True,
                                  truncation=0.7,
                                  truncation_latent=trunc4,
                                  swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
                                  )

                grid = make_grid(torch.cat([imgs_gen1, imgs_gen2, imgs_gen3, imgs_gen4], 0),
                                 nrow=4,
                                 normalize=True,
                                 range=(-1, 1),
                                 )
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                im = pilimg.fromarray(ndarr)
                im.save(f'{outdir}/out-{i*number_of_step+j}.png')
            latent1 = latent2


def create_video(outdir, video_name, number_of_img, number_of_step):
    im = cv2.imread(f'{outdir}/out-0.png')
    frameSize = im.shape[:-1][::-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'{outdir}/{video_name}.mp4', fourcc, 5., frameSize)

    for i in range(number_of_img*(number_of_step)):
        frame = cv2.imread(f'{outdir}/out-{i}.png')
        frame = np.asarray(frame)
        video.write(frame)
    video.release()


def main(args):
    print('Initialization of generators')
    generators = init_generators()
    print('The generators are successfully initialized ')

    print("Starting image generation")
    generate_images(generators, args.number_of_img, args.number_of_step, args.outdir)
    print(f"The images are generated and saved in {args.outdir}")

    print("Starting video creation")
    create_video(args.outdir, args.video_name, args.number_of_img, args.number_of_step)
    print(f"Video saved in {args.outdir}/{args.video_name}.mp4")

    print(f"Finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Images using Pretrained model!")
    parser.add_argument("-o", "--outdir",         type=str, default="results")
    parser.add_argument("-v", "--video_name",     type=str, default="example")
    parser.add_argument("-i", "--number_of_img",  type=int, default=2,)
    parser.add_argument("-s", "--number_of_step", type=int, default=7,)
    args = parser.parse_args()
    main(args)
