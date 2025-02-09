import copy
import torch


def FedAvg(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    # k (string) = conv1.weight, conv1.bias, conv2.weight...
    # i = i-th edge device
    # 把第k個layer中每個device的weight加起來
    for k in w_avg.keys():  
        for i in range(1, len(w)):  
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    if args.gpu : 
        print(f'    Device : GPU')
    else : 
        print(f'    Device : CPU')
    print(f'    Dataset : {args.dataset}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning rate  : {args.lr}')

    print('    Federated parameters:')
    print(f'    Global Rounds   : {args.global_ep}')
    print(f'    Local Epochs       : {args.local_ep}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Number of users   : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}\n')

    



