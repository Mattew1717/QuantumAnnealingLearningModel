import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.multiprocessing as mp
from sklearn.metrics import accuracy_score
from ising_learning_model.data import SimpleDataset, HiddenNodesInitialization, GammaInitialization
from ising_learning_model.sim_anneal_model import SimAnnealModel, AnnealingSettings

# Model parameters
size = 784  # MNIST images are 28x28
batch_size = 80
learning_rate_gamma = 0.25
learning_rate_lmd = 0.0001
lmd = -0.03  
eps = -8
training_epochs = 5
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

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Select only two classes (0 and 1)
mask = (mnist_data.targets == 0) | (mnist_data.targets == 1)
X = mnist_data.data[mask].float().view(-1, 784)  # Flatten images to 784 pixels
y = mnist_data.targets[mask].float()

def main():
    mp.freeze_support()
    print("Model parameters loaded")

    dataset = SimpleDataset()
    test_set = SimpleDataset()

    dataset.x = X[:1000]  # Training set
    dataset.y = y[:1000]
    dataset.len = len(dataset.y)
    dataset.data_size = len(dataset.x[0])
    
    test_set.x = X[1000:1200]  # Test set
    test_set.y = y[1000:1200]
    test_set.len = len(test_set.y)
    test_set.data_size = len(test_set.x[0])

    # Create model
    model = SimAnnealModel(size=size, settings=settings_anneal)
    model.settings.learning_rate_gamma = learning_rate_gamma
    model.settings.learning_rate_lmd = learning_rate_lmd
    model.lmd_init_value = lmd
    model.offset_init_value = eps
    model.settings.optim_steps = training_epochs
    model.settings.mini_batch_size = batch_size

    # Hidden nodes
    model.settings.hidden_nodes_init = HiddenNodesInitialization(hidden_nodes_initial)
    model.settings.hidden_nodes_init.function = SimpleDataset.offset
    model.settings.hidden_nodes_init.fun_args = [1 / size]
    model.settings.gamma_init = GammaInitialization(gamma_initialization)

    # Training
    results = model.train(training_set=dataset, save_params=True, save_samples=False, verbose=False)
    
    # Testing
    r_test = model.test(test_set)
    predictions = r_test.results_samples["energy"].values
    pred = np.where(predictions < ((classes[1]-classes[0])/2), classes[0], classes[1])
    accuracy = accuracy_score(pred, y[1000:1200].cpu())

    print("Test Accuracy:", accuracy)

if __name__ == "__main__":
    main()
