import torch
import torch.nn as nn
import numpy as np
from utils import AnnealingSettings
from data_ import SimpleDataset, HiddenNodesInitialization
from perceptronNet import MultiIsingNetwork
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def fquad(x):
    return  1.2* (x - 0.5) **  2  - 2

def flin(x):
    return 2 * x - 6

FUNC = fquad

NUM_ISING_PERCEPTRONS = 5
SIZE = 50
BATCH_SIZE = 1000
EPOCHS = 200

LAMBDA_INIT = -0.05
OFFSET_INIT = -2.70
LEARNING_RATE_GAMMA = 0.25
LEARNING_RATE_LAMBDA = 0.001
LEARNING_RATE_OFFSET = 0.1
LR_COMBINER = 0.005 # Learning rate for the final linear layer

NUM_SAMPLES_TRAIN = 200
RANGES_TRAIN = [[0, 1]]

NUM_SAMPLES_TEST = 200
RANGES_TEST = [[0, 1]]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# Settings Simulated Annealing
SA_settings = AnnealingSettings()
SA_settings.beta_range = [1, 10]
SA_settings.num_reads = 1
SA_settings.num_sweeps = 1000
SA_settings.sweeps_per_beta = 1

# Hidden nodes parameters
HN_init = "function"
HN_function = SimpleDataset.offset
HN_fun_args = [-1/SIZE * 1]

# === Plot ===
def plotGraphs(x_values, predictions):
    x = np.linspace(RANGES_TEST[0][0], RANGES_TEST[0][1], 200)
    y = FUNC(x)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="target", color='blue')
    plt.scatter(x_values, predictions, color='red', label="predictions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predictions on fquad")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 1. Generazione dei dati
    dataset = SimpleDataset()
    test_set = SimpleDataset()
    dataset.create_data_fun(FUNC, num_samples=NUM_SAMPLES_TRAIN, ranges=RANGES_TRAIN)
    test_set.create_data_fun(FUNC, num_samples=NUM_SAMPLES_TEST, ranges=RANGES_TEST)

    #Hidden nodes creation
    hn = HiddenNodesInitialization("function")
    hn.function = SimpleDataset.offset
    hn.fun_args = [-1/SIZE * 0.8]

    dataset.resize(SIZE, hn)
    dataset.len = len(dataset.y)
    dataset.data_size = len(dataset.x[0])
    
    test_set.resize(SIZE, hn)
    test_set.len = len(test_set.y)
    test_set.data_size = len(test_set.x[0])

    # Conversione a tensori
    X_train = dataset.x.detach().clone().float()
    y_train = dataset.y.detach().clone().float().view(-1)
    X_test = test_set.x.detach().clone().float()
    y_test = test_set.y.detach().clone().float().view(-1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=True)

    model = MultiIsingNetwork(
        num_ising_perceptrons=NUM_ISING_PERCEPTRONS,
        sizeAnnealModel=SIZE,
        anneal_settings=SA_settings,
        lambda_init=LAMBDA_INIT,
        offset_init=OFFSET_INIT,
    ).to(DEVICE)

    # Setup optimizer with per-parameter group learning rates
    optimizer_grouped_parameters = []

    for p_idx, single_module in enumerate(model.ising_perceptrons_layer):
        optimizer_grouped_parameters.append({'params': [single_module.ising_layer.gamma], 'lr': LEARNING_RATE_GAMMA, 'name': f'gamma_{p_idx}'})
        optimizer_grouped_parameters.append({'params': [single_module.lmd], 'lr': LEARNING_RATE_LAMBDA, 'name': f'lambda_{p_idx}'})
        optimizer_grouped_parameters.append({'params': [single_module.offset], 'lr': LEARNING_RATE_OFFSET, 'name': f'offset_{p_idx}'})
    
    # Add parameters for the combiner layer
    optimizer_grouped_parameters.append({'params': model.combiner_layer.parameters(), 'lr': LR_COMBINER, 'name': 'combiner'})
    
    optimizer = torch.optim.SGD(optimizer_grouped_parameters) 
    loss_fn = nn.MSELoss()
   
    # --- Training ---
    model.train_model(train_loader, optimizer, loss_fn, EPOCHS, DEVICE, print_every=10)

    # --- Testing ---
    predictions_test, targets_test = model.test(test_loader, DEVICE)
    mse_on_test = np.mean((predictions_test - targets_test)**2)
    print(f"MSE on test set: {mse_on_test:.4f}")

    x_plot = [e[0] for e in test_set.x]
    plotGraphs(x_plot, predictions_test)



