from torchvision import datasets, transforms
import numpy as np

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == 'MNIST':
        data_dir = '../Dataset/MNIST/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        
    num_items = int(len(train_dataset)/args.num_users)
    user_groups, all_data_idxs = {}, [i for i in range(len(train_dataset))]
    for i in range(args.num_users):
        user_groups[i] = set(np.random.choice(all_data_idxs, num_items,
                                            replace=False))
        all_data_idxs = list(set(all_data_idxs) - user_groups[i])
        
    return train_dataset, test_dataset, user_groups

def get_test_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == 'MNIST':
        data_dir = '../Dataset/MNIST/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        
    return test_dataset