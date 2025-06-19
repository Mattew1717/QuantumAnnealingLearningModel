
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from ising_learning_model.data import SimpleDataset, HiddenNodesInitialization, GammaInitialization
from ising_learning_model.sim_anneal_model import SimAnnealModel, AnnealingSettings
from ising_learning_model.qpu_model import QPUModel
import torch.multiprocessing as mp
from sklearn.metrics import accuracy_score


def plot_loss(losses):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Loss per Epoch", color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.grid()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[-1, 1], yticklabels=[-1, 1])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

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
'''
def generate_xor_balanced(dim, n_samples_dim=1000, shuffle=True):
    
    samples = np.random.random(size=(2**dim*n_samples_dim, dim))
    for i in range(2**dim):
        signs = np.array([1 if int((i // 2**d) % 2) == 0 else -1 for d in range(dim)])
        samples[i*n_samples_dim:(i+1)*n_samples_dim] *= signs
    labels = np.sign(np.prod(samples, axis=1))
    if shuffle:
        perm = np.random.permutation(2**dim*n_samples_dim)
        samples = samples[perm]
        labels = labels[perm]
    return samples, labels
'''
# Model parameters
size = 120
batch_size = 20000
learning_rate_gamma = 0.02
learning_rate_lmd = 0.0001
learning_rate_offset = 1
learning_rate_theta = 1
lmd = -0.03 
eps = -8
training_epochs = 50
classes = [0, 1]
gamma_initialization = "zeros"
hidden_nodes_initial= "function"
random_seed = 42

np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Simulated Annealing settings
settings_anneal = AnnealingSettings()   
settings_anneal.beta_range = [1, 10]
settings_anneal.num_reads = 100
settings_anneal.num_sweeps = 1000
settings_anneal.sweeps_per_beta = 1

def main():
    mp.freeze_support()
    
    dataset = SimpleDataset()
    test_set = SimpleDataset()

    X_train, y_train = generate_xor_balanced(2, n_samples_dim=200, shuffle=True)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_test, y_test = generate_xor_balanced(2, n_samples_dim=100, shuffle=True)
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

    # Create model
    model = SimAnnealModel(size=size, settings=settings_anneal)
    model.settings.learning_rate_gamma = learning_rate_gamma
    model.settings.learning_rate_lmd = learning_rate_lmd
    model.settings.learning_rate_offset = learning_rate_offset
    model.settings.learning_rate_theta = learning_rate_theta
    model.lmd_init_value = lmd
    model.offset_init_value = eps
    model.settings.optim_steps = training_epochs
    model.settings.mini_batch_size = batch_size
    model.settings.optim_steps = training_epochs

    model.settings.hidden_nodes_init = HiddenNodesInitialization(hidden_nodes_initial)
    model.settings.hidden_nodes_init.function = SimpleDataset.offset
    model.settings.hidden_nodes_init.fun_args = [-1/size]
    model.settings.gamma_init = GammaInitialization(gamma_initialization)
    
    # Training
    results = model.train(training_set=dataset, save_params=True, save_samples=False, verbose=False)
    training_time = results.runtime
    losses = results.results["loss"].values
    
    # Testing
    r_test = model.test(test_set)
    predictions = r_test.results_samples["energy"].values
    print(predictions)
    print(y_test.numpy())
    
    pred = np.where(predictions < 0.5, 0, 1)
    
    
    print(pred)
    accuracy = accuracy_score(pred, y_test.numpy())
    plot_confusion_matrix(y_test.numpy(), pred)
    print("Accuracy:", accuracy)
    print("Runtime:", training_time * 60, "s")

    
if __name__ == "__main__":
    main()
