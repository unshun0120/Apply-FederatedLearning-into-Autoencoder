import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import loss_vae

class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]

        return torch.as_tensor(image), torch.as_tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.MSELoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80%, 10%, 10%)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]
        
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test)/10), shuffle=False)
        
        return trainloader, validloader, testloader

    def update_weights(self, args, curr_user, user_idx, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        elif self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)

        print('\nGlobal Round: {}, {}-th Edge device, Local model ID: {}'.format(global_round+1, curr_user, user_idx))
        for local_epoch in range(self.args.local_ep):
            batch_loss = []
            for _, (images, _) in enumerate(tqdm(self.trainloader, desc="Local Round {} ...".format(local_epoch+1))):
                if args.model == 'ae' or args.model == 'vae':
                    # 將(batch_size, channels, height, width)的tensor轉成(batch_size, features)，方便輸入到fully-connected layer
                    images = images.view(images.size(0), -1)
                else:
                    pass

                images = images.to(self.device)
                model.zero_grad()
                if args.model == 'vae' or args.model == 'cnnvae': 
                    s_predicted, mu, logvar = model(images)
                    loss = loss_vae(s_predicted, images, mu, logvar, self.criterion)
                elif args.model == 'vqvae' :
                    s_predicted, vq_loss = model(images)
                    recon_loss = self.criterion(s_predicted, images)
                    loss = recon_loss + vq_loss
                else: 
                    s_predicted = model(images)
                    # get loss value
                    loss = self.criterion(s_predicted, images)

                loss.backward()
                optimizer.step()
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, args, model):
        """ 
        Returns the inference Reconstructino Loss.
        """

        reconstruction_error = 0.0

        with torch.no_grad():
            for _, (images, _) in enumerate(self.testloader):
                if args.model == 'ae' or args.model == 'vae':
                    images = images.view(images.size(0), -1)
                else:
                    pass

                images = images.to(self.device)
                if args.model == 'vae' or args.model == 'cnnvae': 
                    s_predicted, mu, logvar = model(images)
                    loss = loss_vae(s_predicted, images, mu, logvar, self.criterion)
                elif args.model == 'vqvae' :
                    s_predicted, vq_loss = model(images)
                    recon_loss = self.criterion(s_predicted, images)
                    loss = recon_loss + vq_loss
                else: 
                    s_predicted = model(images)
                    # get loss value
                    loss = self.criterion(s_predicted, images)

                reconstruction_error += loss.item()

        reconstruction_error /= len(self.testloader)

        return loss
    
