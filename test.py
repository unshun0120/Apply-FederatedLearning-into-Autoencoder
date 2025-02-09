import torch
import time

import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import autoencoder
from argument import args_parser
from dataset import get_dataset



if __name__ == '__main__': 
    # Start the timer
    start_time = time.time()
    # get argument
    args = args_parser()

    # use GPU or CPU
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'


    train_dataset, test_dataset, user_groups = get_dataset(args)

    # Test
    AE_FL_model = autoencoder().to(device)
    AE_FL_model.load_state_dict(torch.load('../save_models/Autoencoder_MNIST_GE[1]_LE[1]_B[2].pth', weights_only=True))
    AE_FL_model.eval()  # 設置模型為評估模式
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    criterion = nn.MSELoss().to(device)
    reconstruction_error = 0.0

    print("Testing ...")
    with torch.no_grad():
        for _, (images, _) in enumerate(tqdm(testloader, colour="blue")):
            images = images.view(images.size(0), -1)
            images = images.to(device)

            output = AE_FL_model(images)
            loss = criterion(output, images)
            reconstruction_error += loss.item()

    # Computing the Reconstruction Error
    reconstruction_error /= len(testloader)
    print(f"Reconstruction Error: {reconstruction_error}")

    print('\nTotal Run Time: {0:0.4f}'.format(time.time()-start_time))