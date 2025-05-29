import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from utils import utils, SimAnnealModel, AnnealingSettings, QPUModel, ExactModel
import threading

NUM_THREADS = 16

AnnealModel = SimAnnealModel
#Attenzione: se si usa QPUModel, il numero di thread deve essere 1
# AnnealModel = QPUModel
# AnnealModel = ExactModel

# PyTorch Custom Autograd Function
class IsingEnergyFunction(Function):
    @staticmethod
    def forward(ctx, 
                thetas: torch.Tensor, 
                gammas: torch.Tensor, 
                annealModel: AnnealModel):
        
        batch_size, annealingModelSize = thetas.shape
        
        # Converte gamma_torch in numpy e lo imposta nel modello di annealing.
        gamma_np = gammas.detach().cpu().numpy()
        annealModel._gamma = gamma_np

        configurations_bulk = [None] * batch_size
        energies_bulk = [None] * batch_size

        def worker(start_idx, end_idx):
            for i in range(start_idx, end_idx):
                theta = thetas[i].detach().cpu().numpy()
                h = utils.vector_to_biases(theta)
                J = utils.gamma_to_couplings(annealModel._gamma)
                sample_set = annealModel._sample_single(h, J)
                energies_bulk[i] = sample_set.first.energy
                configurations_bulk[i] = list(sample_set.first.sample.values())

        # Divide il batch in blocchi
        chunk_size = (batch_size + NUM_THREADS - 1) // NUM_THREADS
        threads = []

        for t in range(NUM_THREADS):
            start_idx = t * chunk_size
            end_idx = min((t + 1) * chunk_size, batch_size)
            thread = threading.Thread(target=worker, args=(start_idx, end_idx))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()


        energies_bulk = torch.tensor(energies_bulk, dtype=thetas.dtype, device=thetas.device)
        configurations_bulk = np.array(configurations_bulk)
        configurations_bulk = torch.tensor(configurations_bulk, dtype=thetas.dtype, device=thetas.device)

        # Salva z_star per il backward.
        ctx.save_for_backward(configurations_bulk)
        
        return energies_bulk

    @staticmethod
    def backward(ctx, grad_energies_bulk: torch.Tensor):

        configurations_bulk, = ctx.saved_tensors # (batch_size, num_spins)
        batch_size, num_spins = configurations_bulk.shape

        # Calcoliamo dL/dgamma_ij = sum_over_batch( (dL/dE0) * (dE0/dgamma_ij) ) / batch_size
        # Dove dE0/dgamma_ij = z_k_i * z_k_j (configurazione ottimale).
        
        z_i = configurations_bulk.unsqueeze(2) 
        z_j = configurations_bulk.unsqueeze(1) 
        outer_prod_z_star = z_i * z_j

        # grad_gamma_per_sample = dL/dE0 * dE0/dgamma_ij
        grad_gamma_per_sample = grad_energies_bulk.view(batch_size, 1, 1) * outer_prod_z_star
        
        grad_gamma = torch.mean(grad_gamma_per_sample, dim=0)
        
        grad_gamma = utils.make_upper_triangular_torch(grad_gamma)

        return None, grad_gamma, None


# PyTorch nn.Module Layer
class IsingLayer(nn.Module):

    def __init__(self, sizeAnnealModel: int, anneal_settings: AnnealingSettings = AnnealingSettings()):
        super().__init__()
        self.sizeAnnealModel = sizeAnnealModel
        initial_gamma = torch.zeros((sizeAnnealModel, sizeAnnealModel), dtype=torch.float32)
        self.gamma = nn.Parameter(initial_gamma)
        self.annealModel = AnnealModel(sizeAnnealModel, settings=anneal_settings)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        return IsingEnergyFunction.apply(thetas, self.gamma, self.annealModel)

class FullIsingModule(nn.Module):
    def __init__(self, sizeAnnealModel: int, 
                 anneal_settings: AnnealingSettings = AnnealingSettings(),
                 lambda_init: float = 1.0, 
                 offset_init: float = 0.0):
        super().__init__()
        self.sizeAnnealModel = sizeAnnealModel
        self.ising_layer = IsingLayer(sizeAnnealModel, anneal_settings)
        
        # Parametri learnable
        self.lmd = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))
        self.offset = nn.Parameter(torch.tensor(offset_init, dtype=torch.float32))

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        E0_batch = self.ising_layer(thetas)
        output = self.lmd * E0_batch + self.offset
        return output