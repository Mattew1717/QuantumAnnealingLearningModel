import torch
import torch.nn as nn
import numpy as np
from utils import AnnealingSettings
from data_ import SimpleDataset, HiddenNodesInitialization
from perceptronNet import MultiIsingNetwork
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import time

#PATH = "C:/Users/Matteo/Documents/GitHub/IsingModel/Ising_Learning_Model/ising-learning-model-main/datasets/breast-cancer.csv"

def generate_xor_balanced(dim, n_samples_dim=1000, shuffle=True):
    '''
    Generate XOR data in U[-1,1]^d with balanced classes.

    :param dim(int): dimension of the XOR problem
    :param n_samples_dim(int, optional): number of samples in each region
    :param shuffle(bool, optional): shuffle the data

    :returns: generated samples and labels
    '''
    samples = np.random.random(size=(2**dim*n_samples_dim, dim))
    for i in range(2**dim):
        signs = np.array([1 if int((i // 2**d) % 2) == 0 else -1 for d in range(dim)])
        samples[i*n_samples_dim:(i+1)*n_samples_dim] *= signs
    labels = np.sign(np.prod(samples, axis=1))
    
    if shuffle:
        perm = np.random.permutation(2**dim*n_samples_dim)
        samples = samples[perm]
        labels = labels[perm]
    labels = np.where(labels < 0, 0, 1)
    return samples, labels

PARTITION_INPUT = False
NUM_ISING_PERCEPTRONS = 10
SIZE = 30
BATCH_SIZE = 1000
EPOCHS = 100

LAMBDA_INIT = -0.03
OFFSET_INIT = -8
LEARNING_RATE_GAMMA = 0.1
LEARNING_RATE_LAMBDA = 0.1
LEARNING_RATE_OFFSET = 0.1
LR_COMBINER = 0.01 # Learning rate for the final linear layer

PERCENT_TRAIN = 0.8
PERCENT_TEST = 1 - PERCENT_TRAIN
CLASSES = [0, 1]

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

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[CLASSES[0], CLASSES[1]], yticklabels=[CLASSES[0], CLASSES[1]])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    torch.set_num_threads(os.cpu_count())

    # Load dataset
    #data = pd.read_csv(PATH)
    #data.iloc[:, -1] = data.iloc[:, -1].replace({-1: CLASSES[0], 1: CLASSES[1]})

    dataset = SimpleDataset()
    test_set = SimpleDataset()

    X_train, y_train = generate_xor_balanced(3, n_samples_dim=200, shuffle=True)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_test, y_test = generate_xor_balanced(3, n_samples_dim=100, shuffle=True)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    

    dataset.x = X_train
    dataset.y = y_train
    dataset.data_size = len(dataset.x[0])
    dataset.len = len(dataset.y)

    test_set.x = X_test
    test_set.y = y_test
    test_set.data_size = len(test_set.x[0])
    test_set.len = len(test_set.y)

    #Hidden nodes definition
    hn = HiddenNodesInitialization(HN_init)
    hn.function = HN_function
    hn.fun_args = HN_fun_args

    if PARTITION_INPUT:
        dataset.resize(SIZE*NUM_ISING_PERCEPTRONS, hn)
        dataset.len = len(dataset.y)
        dataset.data_size = len(dataset.x[0])

        test_set.resize(SIZE*NUM_ISING_PERCEPTRONS, hn)
        test_set.len = len(test_set.y)
        test_set.data_size = len(test_set.x[0])
    else:
        dataset.resize(SIZE, hn)
        dataset.len = len(dataset.y)
        dataset.data_size = len(dataset.x[0])

        test_set.resize(SIZE, hn)
        test_set.len = len(test_set.y)
        test_set.data_size = len(test_set.x[0])

    train_loader = DataLoader(TensorDataset(dataset.x, dataset.y), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(TensorDataset(test_set.x, test_set.y), batch_size=BATCH_SIZE, shuffle=False)

    model = MultiIsingNetwork(
        num_ising_perceptrons=NUM_ISING_PERCEPTRONS,
        sizeAnnealModel=SIZE,
        anneal_settings=SA_settings,
        lambda_init=LAMBDA_INIT,
        offset_init=OFFSET_INIT,
        partition_input=PARTITION_INPUT
    ).to(DEVICE)

    # Setup optimizer with per-parameter group learning rates
    optimizer_grouped_parameters = []

    for p_idx, single_module in enumerate(model.ising_perceptrons_layer):
        optimizer_grouped_parameters.append({'params': [single_module.ising_layer.gamma], 'lr': LEARNING_RATE_GAMMA, 'name': f'gamma_{p_idx}'})
        optimizer_grouped_parameters.append({'params': [single_module.lmd], 'lr': LEARNING_RATE_LAMBDA, 'name': f'lambda_{p_idx}'})
        optimizer_grouped_parameters.append({'params': [single_module.offset], 'lr': LEARNING_RATE_OFFSET, 'name': f'offset_{p_idx}'})
    
    # Add parameters for the combiner layer
    optimizer_grouped_parameters.append({'params': model.combiner_layer.parameters(), 'lr': LR_COMBINER, 'name': 'combiner'})
    
    optimizer = torch.optim.Adam(optimizer_grouped_parameters) 
    loss_fn = nn.BCEWithLogitsLoss()
    start_time = time.time()
    # --- Training ---
    model.train_model(train_loader, optimizer, loss_fn, EPOCHS, DEVICE, print_every=1)
    end_time = time.time()
    
    # --- Testing ---
    predictions_test, targets_test = model.test(test_loader, DEVICE)
    mse_on_test = np.mean((predictions_test - targets_test)**2)
    print(f"MSE on test set: {mse_on_test:.4f}")

    predictions = np.where(predictions_test < (abs(CLASSES[1])-abs(CLASSES[0]))/2, CLASSES[0], CLASSES[1]) #classe0 deve essere piu piccola di classe1

    errors = predictions != targets_test
    errors_class_neg1 = np.sum((targets_test == CLASSES[0]) & errors)
    errors_class_1 = np.sum((targets_test == CLASSES[1]) & errors)
    accuracy = accuracy_score(predictions, targets_test)
    plot_confusion_matrix(test_set.y.numpy(), predictions)

    print("predictions: ", predictions)
    print("targets: ", targets_test)
    print("Accuracy:", accuracy)
    print(f"Errori classe {CLASSES[0]}: ", errors_class_neg1)
    print(f"Errori classe {CLASSES[1]}: ", errors_class_1)
    print(f"Training time: {end_time - start_time:.2f} seconds")

