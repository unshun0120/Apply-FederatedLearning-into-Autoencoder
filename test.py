import torch
import time
import os

import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.AE import autoencoder, cnn_autoencoder
from model.VAE import vae, cnn_vae
from model.VQ_VAE import vqvae
from argument import args_parser
from dataset import get_test_dataset
from utils import loss_vae



if __name__ == '__main__': 
    # Start the timer
    start_time = time.time()

    # get argument
    args = args_parser()

    # use GPU or CPU
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    file_name = '../save_objects/Autoencoder_{}_GE[{}]_LE[{}]_B[{}].pkl'.\
            format(args.dataset, args.global_ep, args.local_ep, args.local_bs)
    # Test
    dataset = 'MNIST'
    global_ep = 1
    local_ep = 1
    local_bs = 2

    if args.model == 'ae': 
        model_file_path = '../save_models/Autoencoder_{}_GE[{}]_LE[{}]_B[{}].pth'.\
            format(dataset, global_ep, local_ep, local_bs)
        AE_FL_model = autoencoder().to(device)
    elif args.model == 'cnnae':
        model_file_path = '../save_models/CNN_Autoencoder_{}_GE[{}]_LE[{}]_B[{}].pth'.\
            format(dataset, global_ep, local_ep, local_bs)
        AE_FL_model = cnn_autoencoder().to(device)
    elif args.model == 'vae':
        model_file_path = '../save_models/VAE_{}_GE[{}]_LE[{}]_B[{}].pth'.\
            format(dataset, global_ep, local_ep, local_bs)
        AE_FL_model = vae().to(device)
    elif args.model == 'cnnvae':
        model_file_path = '../save_models/CNNVAE_{}_GE[{}]_LE[{}]_B[{}].pth'.\
            format(dataset, global_ep, local_ep, local_bs)
        AE_FL_model = cnn_vae().to(device)
    elif args.model == 'vqvae':
        model_file_path = '../save_models/VQVAE_{}_GE[{}]_LE[{}]_B[{}].pth'.\
            format(dataset, global_ep, local_ep, local_bs)
        AE_FL_model = vqvae().to(device)
    
    AE_FL_model.load_state_dict(torch.load(model_file_path, weights_only=True))
    AE_FL_model.eval()  

    test_dataset = get_test_dataset(args)
    test_batch_size = 32
    testloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    criterion = nn.MSELoss().to(device)
    reconstruction_error = 0.0

    print("Testing ...")
    with torch.no_grad():
        for _, (images, labels) in enumerate(tqdm(testloader, colour="blue")):
            if args.model == 'ae' or args.model == 'vae':
                images = images.view(images.size(0), -1)
            else:
                pass

            images = images.to(device)
            if args.model == 'vae' or args.model == 'cnnvae': 
                s_predicted, mu, logvar = AE_FL_model(images)
                loss = loss_vae(s_predicted, images, mu, logvar, criterion)
            elif args.model == 'vqvae' :
                s_predicted, vq_loss = AE_FL_model(images)
                recon_loss = criterion(s_predicted, images)
                loss = recon_loss + vq_loss
            elif args.model == 'ae' or args.model == 'cnnae': 
                s_predicted = AE_FL_model(images)
                loss = criterion(s_predicted, images)

            reconstruction_error += loss.item()

    # Computing the Reconstruction Error
    reconstruction_error /= len(testloader)
    print(f"Reconstruction Error: {reconstruction_error}")

    print("Generating Test Images ...")

    if args.model == 'ae': 
        output_dir = './output_img/Autoencoder_{}_GE[{}]_LE[{}]_B[{}]'.\
            format(dataset, global_ep, local_ep, local_bs)
    elif args.model == 'cnnae':
        output_dir = './output_img/CNN_Autoencoder_{}_GE[{}]_LE[{}]_B[{}]'.\
            format(dataset, global_ep, local_ep, local_bs)
    elif args.model == 'vae':
        output_dir = './output_img/VAE_{}_GE[{}]_LE[{}]_B[{}]'.\
            format(dataset, global_ep, local_ep, local_bs)
    elif args.model == 'cnnvae':
        output_dir = './output_img/CNNVAE_{}_GE[{}]_LE[{}]_B[{}]'.\
            format(dataset, global_ep, local_ep, local_bs)
    elif args.model == 'vqvae':
        output_dir = './output_img/VQVAE_{}_GE[{}]_LE[{}]_B[{}]'.\
            format(dataset, global_ep, local_ep, local_bs)

    os.makedirs(output_dir, exist_ok=True)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    saved_digits = []

    with torch.no_grad():
        for _, (images, labels) in enumerate(tqdm(testloader, colour="blue")):
            if args.model == 'ae' or args.model == 'vae':
                images = images.view(images.size(0), -1)
            else:
                pass

            images = images.to(device)
            if args.model == 'vae' or args.model == 'cnnvae': 
                s_predicted, mu, logvar = AE_FL_model(images)
            elif args.model == 'vqvae' :
                s_predicted, vq_loss = AE_FL_model(images)
            elif args.model == 'ae' or args.model == 'cnnae': 
                s_predicted = AE_FL_model(images)

            digit = labels
            if digit not in saved_digits:
                saved_digits.append(digit)
                # [1, 784] -> [1, 1, 28, 28]
                images = images.view(images.size(0), 1, 28, 28) 
                s_predicted = s_predicted.view(s_predicted.size(0), 1, 28, 28)

                fig, axes = plt.subplots(1, 2, figsize=(6, 3))
                axes[0].imshow(images.cpu().squeeze().numpy(), cmap='gray')
                axes[0].set_title(f'Original {digit}')
                axes[1].imshow(s_predicted.cpu().squeeze().numpy(), cmap='gray')
                axes[1].set_title(f'Reconstructed {digit}')
                
                for ax in axes:
                    ax.axis('off')
                
                plt.savefig(os.path.join(output_dir, f'{digit}.png'))
                plt.close()
                
            if len(saved_digits) >= 10:
                break

    print("Images(0~9) have been saved in :", output_dir, " folder")



    print('\nTotal Run Time: {0:0.4f}'.format(time.time()-start_time))