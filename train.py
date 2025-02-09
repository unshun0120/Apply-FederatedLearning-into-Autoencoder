import torch
import numpy as np
import copy
import pickle
import time

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from argument import args_parser
from dataset import get_dataset
from model import autoencoder
from local_model import LocalUpdate
from utils import FedAvg, exp_details


if __name__ == '__main__':
    start_time = time.time()

    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # use GPU or CPU
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    # Load dataset
    print(f'Loading {args.dataset} dataset ...\n')
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # Autoencoder Global model
    global_model = autoencoder()
    global_model.to(device)
    global_model.train()

    global_weights = global_model.state_dict()
    
    # Training
    train_loss = []

    for epoch in range(args.global_ep): 
        local_weights, local_losses = [], []
        # How many edge devices have run so far
        curr_user = 1   

        global_model.train()

        # compute the number of clients that the server needs to send the global model
        """
        in ref. paper : "Communication-Efficient Learning of Deep Networks from Decentralized Data" 中提到
        At the beginning of each round, a random fraction of clients is selected, and the server 
        sends the current global algorithm state to each of these clients (e.g., the current model parameters). 
        We only select a fraction of clients for efficiency
        計算 m = frac*user, 從所有的user中random挑出m個user
        """
        # defaulted m : 0.1 * 100 = 10
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # local model training
        for idx in idxs_users: 
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(curr_user, idx, model=copy.deepcopy(global_model), global_round=epoch)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            curr_user = curr_user + 1


        # Using "Federated Average" to update global weight strategy
        global_weights = FedAvg(local_weights)

        # update global model
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Local model Inference
        inference_loss = []
        global_model.eval()
        print(f'\nGlobal Round: {format(epoch+1)} - Local model inference ...')
        for c in tqdm(range(args.num_users), colour="yellow"):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            loss = local_model.inference(model=global_model)
            inference_loss.append(loss)
            
        print(f'\nAvg Training Stats after {epoch+1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print(f'Inference Loss : {sum(inference_loss)/len(inference_loss)}\n')

    # Saving the training loss objects:
    print("Saving the Autoencoder model training loss objects...")
    file_name = '../save_objects/Autoencoder_{}_GE[{}]_LE[{}]_B[{}].pkl'.\
        format(args.dataset, args.global_ep, args.local_ep, args.local_bs)
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss], f)

    #Save the model
    print("Saving the Autoencoder model ...\n")
    torch.save(global_model.state_dict(), '../save_models/Autoencoder_{}_GE[{}]_LE[{}]_B[{}].pth'.\
        format(args.dataset, args.global_ep, args.local_ep, args.local_bs))
    print("Saving Complete !!!")

    print('\nTotal Run Time: {0:0.4f}'.format(time.time()-start_time))