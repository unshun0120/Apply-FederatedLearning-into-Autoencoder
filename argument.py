import argparse

# input arguments to set the parameters value
def args_parser():
    parser = argparse.ArgumentParser()

    # environment setup
    # GPU or CPU
    parser.add_argument('--gpu', default=None, help="To use cuda, set to a specific GPU ID. Default set to use CPU.")

    # model something setup
    # Autoencoder Model
    parser.add_argument('--model', type=str, default='ae', help="name of autoencoder or its variants. Default set to use Autoencoder")
    # Dataset
    parser.add_argument('--dataset', type=str, default='MNIST', help="name of dataset. Default set to use CIFAR10")
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help="type of optimizer")
    # Learning rate
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    # Federated Learning setup 
    # Epochs
    parser.add_argument('--global_ep', type=int, default=1, help="number of global model training rounds (epochs)")
    # number of edge devices (i.e. users, clients)
    parser.add_argument('--num_users', type=int, default=100, help="number of edge devices (users): K")
    # Fraction of selected edge devices
    parser.add_argument('--frac', type=float, default=0.1, help='the fraction of clients: C')
    # Local model epochs
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    # Local model batch size
    parser.add_argument('--local_bs', type=int, default=2, help="local batch size: B")

    args = parser.parse_args()
    return args

