import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, BatchSampler
from rexplain.torch import MaskLayer1d
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np


def train_surr(X_train, X_val, original_model, num_genes):

    # Custom loss function
    class CELoss(nn.Module):
        '''Cross entropy loss for distributional targets. Expects logits.'''
        def __init__(self):
            super().__init__()

        def forward(self, pred, target):
            return - torch.mean(torch.sum(pred.ms(dim=1) * target, dim=1))
        
    # Prepare device
    device = torch.device('cuda')

    # Prepare training data
    X_model = torch.tensor(X_train, dtype=torch.float32)
    Y_model = torch.tensor(original_model.predict(X_train),dtype=torch.float32)
    # Prepare validation data
    Y_val_surrogate = torch.tensor(original_model.predict(X_val).repeat(1000, 0),dtype=torch.float32)
    X_val_surrogate = torch.tensor(X_val.repeat(1000, 0), dtype=torch.float32)

    # Random subsets
    S_val = torch.ones(X_val_surrogate.shape)
    num_included = np.random.choice(num_genes + 1, size=len(S_val))
    for i in range(len(S_val)):
        S_val[i, num_included[i]:] = 0
        S_val[i] = S_val[i, torch.randperm(num_genes)]

    # Create dataset iterator
    val_set = TensorDataset(X_val_surrogate, Y_val_surrogate, S_val)
    val_loader = DataLoader(val_set, batch_size=25000)

    def validate(model):
        '''Measure performance on validation set.'''
        with torch.no_grad():
            # Setup
            mean_loss = 0
            N = 0

            # Iterate over validation set
            for x, y, S in val_loader:
                x = x.to(device)
                y = y.to(device)
                S = S.to(device)
                pred = model((x, S))
                loss = loss_fn(pred, y)
                N += len(x)
                mean_loss += len(x) * (loss - mean_loss) / N

        return mean_loss
    # Set up model
    model = nn.Sequential(
        MaskLayer1d(value=0),
        nn.Linear(2 * num_genes, 64),
        nn.ELU(inplace=True),
        nn.Linear(64, 64),
        nn.ELU(inplace=True),
        nn.Linear(64, 4)).to(device)

    # Training parameters
    lr = 1e-3
    nepochs = 1000
    early_stop_epochs = 10

    # Loss function
    loss_fn = CELoss()
    loss_list = []

    for mbsize in (32, 128, 512, 1024, 2048, 5096, 10192):
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Set up data loaders
        train_set = TensorDataset(X_model, Y_model)
        random_sampler = RandomSampler(
            train_set, replacement=True,
            num_samples=int(np.ceil(len(X_model) / mbsize))*mbsize)
        batch_sampler = BatchSampler(random_sampler, batch_size=mbsize, drop_last=True)
        train_loader = DataLoader(train_set, batch_sampler=batch_sampler)

        # For saving best model
        min_epoch = 0
        best_model = deepcopy(model)
        best_epoch_loss = validate(model).item()
        best_epoch = 0

        # Begin training
        for epoch in range(nepochs):
            for i, (x, y) in enumerate(train_loader):
                # Prepare data
                x = x.to(device)
                y = y.to(device)

                # Generate subset
                S = torch.ones(mbsize, num_genes, dtype=torch.float32, device=device)
                num_included = np.random.choice(num_genes + 1, size=mbsize)
                for j in range(mbsize):
                    S[j, num_included[j]:] = 0
                    S[j] = S[j, torch.randperm(num_genes)]

                # Make predictions
                pred = model((x, S))
                loss = loss_fn(pred, y)

                # Optimizer step
                loss.backward()
                optimizer.step()
                model.zero_grad()

            # End of epoch progress message
            val_loss = validate(model).item()
            loss_list.append(val_loss)
            print('----- Epoch = {} -----'.format(epoch + 1))
            print('Val loss = {:.4f}'.format(val_loss))
            print('')

            # Check if best model
            if epoch >= min_epoch:
                if val_loss < best_epoch_loss:
                    best_epoch_loss = val_loss
                    best_model = deepcopy(model)
                    best_epoch = epoch
                    print('New best epoch, val loss = {:.4f}'.format(val_loss))
                    print('')
                else:
                    # Check for early stopping
                    if epoch - best_epoch == early_stop_epochs:
                        print('Stopping early')
                        break

        model = best_model

        # Make model callable with numpy input
        model_lam = lambda x, S: torch.softmax(
        model((torch.tensor(x, dtype=torch.float32, device=device),
            torch.tensor(S, dtype=torch.float32, device=device))),
        dim=1).cpu().data.numpy()

        return model_lam
