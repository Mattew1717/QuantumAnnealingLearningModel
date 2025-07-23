import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils import AnnealingSettings
from data_ import SimpleDataset,HiddenNodesInitialization
from torch.utils.data import TensorDataset, DataLoader
from IsingModule import FullIsingModule


def fquad(x):
    return  1.2* (x - 0.5) **  2  - 2

def flin(x):
    return 2 * x - 6

def flog(x):
    return np.log(x)

def fEmil(x):
    return 0.5*x if x<0 else np.log(x+1) 

FUNC = fEmil

SIZE = 20
BATCH_SIZE = 50
EPOCHS = 2

LAMBDA_INIT = 1
OFFSET_INIT = -2.4
LEARNING_RATE_GAMMA = 0.25
LEARNING_RATE_LAMBDA = 1e-5
LEARNING_RATE_OFFSET = 0.01

NUM_SAMPLES_TRAIN = 50
RANGES_TRAIN = [[-1, 1]]

NUM_SAMPLES_TEST = 200
RANGES_TEST = [[-2, 2]]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

SA_settings = AnnealingSettings()
SA_settings.beta_range = [1, 10]
SA_settings.num_reads = 1
SA_settings.num_sweeps = 1000
SA_settings.sweeps_per_beta = 1


def main():

    # 1. Dataset creation
    torch.manual_seed(RANDOM_SEED)
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

    # Conversion to tensors
    X_train = dataset.x.detach().clone().float()
    y_train = dataset.y.detach().clone().float().view(-1)
    X_test = test_set.x.detach().clone().float()
    y_test = test_set.y.detach().clone().float().view(-1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    # 2. Modello
    model = FullIsingModule(SIZE, SA_settings, LAMBDA_INIT, OFFSET_INIT).to(DEVICE)
    optimizer = torch.optim.SGD([
        {'params': [model.ising_layer.gamma], 'lr': LEARNING_RATE_GAMMA},
        {'params': [model.lmd], 'lr': LEARNING_RATE_LAMBDA},
        {'params': [model.offset], 'lr': LEARNING_RATE_OFFSET},
    ])  
    loss_fn = nn.MSELoss()

    # 3. Training
    for epoch in range(EPOCHS):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f}")

    # 4. Test
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(DEVICE)).cpu().numpy()

    x_plot = [e[0] for e in test_set.x]
    plotGraphs(x_plot, preds)
    print("Final lambda:", model.lmd.item())
    print("Final offset:", model.offset.item())
    print("||gamma|| =", model.ising_layer.gamma.norm().item())

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


if __name__ == "__main__":
    main()