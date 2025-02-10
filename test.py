import torch
import time
import random
import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import autoencoder
from argument import args_parser
from dataset import get_test_dataset



if __name__ == '__main__': 
    # Start the timer
    start_time = time.time()

    # get argument
    args = args_parser()

    # set seeds (每次執行時，隨機過程都會依照相同的模式進行)
    seed = 40
    # Python built-in random module 
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if args.gpu:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # use GPU or CPU
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    # Test
    AE_FL_model = autoencoder().to(device)
    AE_FL_model.load_state_dict(torch.load('../save_models/Autoencoder_MNIST_GE[2]_LE[1]_B[2].pth', weights_only=True))
    AE_FL_model.eval()  

    test_dataset = get_test_dataset(args)
    test_batch_size = 32
    testloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    criterion = nn.MSELoss().to(device)
    reconstruction_error = 0.0

    print("Testing ...")
    
    with torch.no_grad():
        for _, (images, labels) in enumerate(tqdm(testloader, colour="blue")):
            images = images.view(images.size(0), -1)
            images = images.to(device)
            
            output = AE_FL_model(images)
            loss = criterion(output, images)
            reconstruction_error += loss.item()
    

    """     
    images = images[0].cpu()
    # outpt.size() = [16, 784]
    output = output[0].cpu()   
    # outpt.size() = [784] 
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(images.view(28, 28).numpy(), cmap='gray')
    axes[0].set_title('Original Image')
    # 將output tensor([784])轉成照片格式(28 * 28)
    # .numpy(): imshow是處理numpy array
    axes[1].imshow(output.view(28, 28).numpy(), cmap='gray')
    axes[1].set_title('Reconstructed Image')
    plt.axis('off')  # 隱藏軸
    
    plt.savefig('../save_images/output_image.png')
    plt.show()
    """
    # Computing the Reconstruction Error
    reconstruction_error /= len(testloader)
    print(f"Reconstruction Error: {reconstruction_error}")

    print('\nTotal Run Time: {0:0.4f}'.format(time.time()-start_time))