import os, math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import time
from experiments.datasets import *
from utils.func import *

class MaterializedDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.dataset = data
    
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx: int):
        return self.dataset[idx]

def negative_log_likelihood(dataset, logweights):
    logweights = torch.as_tensor(np.array(logweights))
    data = torch.as_tensor(dataset)

    i = data[:, 0].long()
    j = data[:, 1].long()
    m1 = data[:, 2].to(torch.float)
    m2 = data[:, 3].to(torch.float)

    w_i = logweights[i]
    w_j = logweights[j]
    diff = w_i - w_j  

    log_pi = F.logsigmoid(diff)       
    log_pj = F.logsigmoid(-diff)      

    # Binomial log-likelihood with counts
    # We normalize by total games in the batch so the loss scale is stable
    total_games = (m1 + m2).clamp_min(1.0).sum()  # avoid div-by-zero in degenerate batch
    #nll = -(m1 * log_pi + m2 * log_pj).sum() / total_games
    nll = -(m1 * log_pi + m2 * log_pj).sum() 

    return nll.item()

class LightningMNL(pl.LightningModule):
    def __init__(
        self,
        num_items: int,
        lr: float = 0.001,
        initial_params = None,
        alpha: float = 0.01
    ):
        super().__init__()
        self.save_hyperparameters()

        if initial_params is None:
            initial_params = np.zeros(num_items)
        self.logweights = nn.Parameter(torch.tensor(initial_params, dtype=torch.float32))
        
        self.lr = lr
        self.alpha = alpha

        self.last_weights = None
        self.num_items = num_items


    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Remove the unidentifiable degree of freedom: enforce mean(w)=0
        with torch.no_grad():
            self.logweights -= self.logweights.mean()
    
    #@torch.no_grad()
    #def on_train_epoch_end(self):
    #    with torch.no_grad():
    #        self.logweights -= self.logweights.mean()
    #        ell1 = float(self.num_items)
    #        if self.last_weights is not None:
    #            ell1 = torch.linalg.norm(self.last_weights - self.logweights, ord=1)
    #        #print('epoch:', self.current_epoch, 'ended with error:', ell1)
    #        self.log("ell1-norm-error", ell1, prog_bar=True, on_epoch=True, on_step=False)
    #        self.last_weights = self.logweights.detach().clone()

    def configure_optimizers(self):
        #optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {
            "optimizer": optimizer,
        }

    @staticmethod
    def _batch_from_tensor(batch: torch.Tensor):
        """
        Accepts a tensor of shape (B,4) with columns: i, j, m1, m2
        Returns (i, j, m1, m2) as long/float tensors on the correct device.
        """
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)
        i = batch[:, 0].long()
        j = batch[:, 1].long()
        m1 = batch[:, 2].to(torch.float)
        m2 = batch[:, 3].to(torch.float)
        return i, j, m1, m2

    def forward(self, i:torch.Tensor, j:torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logweights[i] - self.logweights[j])

    def training_step(self, batch, batch_idx):
        # Collate gives torch.Tensor from our custom collate_fn below.
        i, j, m1, m2 = self._batch_from_tensor(batch)
        m1 += self.alpha
        m2 += self.alpha

        w_i = self.logweights[i]
        w_j = self.logweights[j]
        diff = w_i - w_j  

        log_pi = F.logsigmoid(diff)       
        log_pj = F.logsigmoid(-diff)      

        # Binomial log-likelihood with counts
        # We normalize by total games in the batch so the loss scale is stable
        total_games = (m1 + m2).clamp_min(1.0).sum()  # avoid div-by-zero in degenerate batch
        nll = -(m1 * log_pi + m2 * log_pj).sum() / total_games

        #nll += self.alpha * torch.sum(self.logweights ** 2) #add squared ell2 norm as penalty

        self.log("train_nll", nll, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return nll

    @torch.no_grad()
    def predict_proba(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        return self(i, j)

def SGD(n, data, seed, batch_size=2048, max_epochs = 100, alpha=0.01,
        lr:float = 0.0001, early_stopping: bool = True, initial_params=None):

    pl.seed_everything(seed)

    dataset = MaterializedDataset(data)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,    #4
        prefetch_factor=4, #2
        persistent_workers=True,
        shuffle=True,
    )

    current_file_dir = os.path.dirname(__file__)  # folder where this script is located
    pathLog = os.path.abspath(os.path.join(current_file_dir, '..', '..', 'lightning_logger'))
    os.makedirs(pathLog, exist_ok=True)
    
    callbacks = []
    if early_stopping:
        early_stop = EarlyStopping(
            monitor="train_nll",
            min_delta=1e-7, 
            patience=2,
            mode="min",
            check_on_train_epoch_end=True,
            verbose=False,
        )
        callbacks.append(early_stop)
    
    trainer = pl.Trainer(            
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        default_root_dir=pathLog,
        callbacks=callbacks,
    )

    model = LightningMNL(n, lr=lr, initial_params=initial_params, alpha=alpha)
    trainer.fit(model, loader)

    w_list = model.logweights.detach().cpu().tolist()
    return MNL(None, w_list, logweights=True), trainer
