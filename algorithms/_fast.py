import torch
import torch.nn as nn
from fastshap.utils import MaskLayer1d
from fastshap import Surrogate, KLDivLoss
from algorithms._explainer import Explainer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from fastshap import FastSHAP
import time 
from tqdm import tqdm

class Fast(Explainer):
    def __init__(self, method, dname, model, num_features):
        self.model = model
        self.num_features = num_features
        self.dname = dname
        self.method = method


    def getfastshapmodel(self, X_train, X_val, num_features):
        # Select device
        device = torch.device('cuda')
        torch.set_num_threads(1)
        # Create surrogate model
        surr = nn.Sequential(
            MaskLayer1d(value=0, append=True),
            nn.Linear(2 * num_features, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 2)).to(device)

        # Set up surrogate object
        surrogate = Surrogate(surr, num_features)

        # Set up original model
        def original_model(x):
            pred = self.model.predict(x.cpu().numpy())
            pred = np.stack([1 - pred, pred]).T
            return torch.tensor(pred, dtype=torch.float32, device=x.device)
        
        # Train
        surrogate.train_original_model(
            X_train,
            X_val,
            original_model,
            batch_size=64,
            max_epochs=100,
            loss_fn=KLDivLoss(),
            validation_samples=10,
            validation_batch_size=10000,
            verbose=True)

        # Create explainer model
        explainer = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2 * num_features)).to(device)

        # Set up FastSHAP object
        fastshap = FastSHAP(explainer, surrogate, normalization='additive',link=nn.Softmax(dim=-1))

        # Train
        fastshap.train(
            X_train,
            X_val[:100],
            batch_size=32,
            num_samples=32,
            max_epochs=200,
            validation_samples=128,
            verbose=True)
        

        return fastshap


    def _explain(self, X_train, y_train, X_test, y_test, X_val):
        start1 = time.time()
        fastshap = self.getfastshapmodel(X_train, X_val, self.num_features)
        phis = []
        X_mod = list()
        start2 = time.time()

        for ind in tqdm(range(X_test.shape[0])):
            x = X_test[ind:ind+1]
            y = int(y_test[ind])
            # Run FastSHAP
            fastshap_values = fastshap.shap_values(x)[0]
            # print(x.shape, fastshap_values.shape, y)
            # print(x.shape, fastshap_values)
            if y < fastshap_values.shape[1]:
                phis.append(fastshap_values[:, y])
                X_mod.append(x[0])

        end = time.time()
        X_test_mod = np.array(X_mod)
        print(X_test_mod.shape)
        attributions = {
            'shap_values': np.array(phis),
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test_mod,
            'y_test': y_test,
            'pred': self.model.predict(X_test_mod),
            'compute_time': end-start1,
            'compute_time_given_models': end-start2
        }

        return attributions

    def __call__(self,X_train, y_train, X_test, y_test, X_val):
        return self._explain(X_train, y_train, X_test, y_test, X_val)