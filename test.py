import torch
import time

import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.AE import autoencoder, cnn_autoencoder
from model.VAE import vae, cnn_vae
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
            else: 
                s_predicted = AE_FL_model(images)
                # get loss value
                loss = criterion(s_predicted, images)

            reconstruction_error += loss.item()

    # Computing the Reconstruction Error
    reconstruction_error /= len(testloader)
    print(f"Reconstruction Error: {reconstruction_error}")

    print('\nTotal Run Time: {0:0.4f}'.format(time.time()-start_time))