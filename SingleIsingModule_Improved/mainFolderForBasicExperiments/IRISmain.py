import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ising_learning_model.data import SimpleDataset, HiddenNodesInitialization, GammaInitialization
from ising_learning_model.sim_anneal_model import SimAnnealModel, AnnealingSettings
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[-1, 1], yticklabels=[-1, 1])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# Model parameters
size = 50   
batch_size = 1000
learning_rate_gamma = 0.02
learning_rate_lmd = 0.000001
learning_rate_offset = 0.1
learning_rate_theta = 0
lmd = -0.03  
eps = -8
training_epochs = 200
classes = [0, 1]
gamma_initialization = "zeros"
hidden_nodes_initial= "function"
random_seed = 42
 
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Simulated Annealing settings
settings_anneal = AnnealingSettings()
settings_anneal.beta_range = [1, 10]
settings_anneal.num_reads = 1
settings_anneal.num_sweeps = 1000
settings_anneal.sweeps_per_beta = 1

# Load dataset
data = pd.read_csv("")
data.iloc[:, -1] = data.iloc[:, -1].replace({-1: classes[0], 1: classes[1]})

# Shuffle rows
data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
X = torch.tensor(data.iloc[:, :-1].values.astype(float), dtype=torch.float32)
y = torch.tensor(data.iloc[:, -1].values.astype(float), dtype=torch.float32)

def main():
    
    dataset = SimpleDataset()
    test_set = SimpleDataset()

    dataset.x = X[:100]
    dataset.y = y[:100]
    dataset.len = len(dataset.y)
    dataset.data_size = len(dataset.x[0])
    
    test_set.x = X[100:]
    test_set.y = y[100:]
    test_set.len = len(test_set.y)
    test_set.data_size = len(test_set.x[0])

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
    

    model.settings.hidden_nodes_init = HiddenNodesInitialization(hidden_nodes_initial)
    model.settings.hidden_nodes_init.function = SimpleDataset.offset
    model.settings.hidden_nodes_init.fun_args = [-1/size * 1]

    model.settings.gamma_init = GammaInitialization(gamma_initialization)

    # Training
    results = model.train(training_set=dataset, save_params=True, save_samples=False, verbose=False)
    training_time = results.runtime
    
    # Testing
    r_test = model.test(test_set)
    predictions = r_test.results_samples["energy"].values
    pred = np.where(predictions < 0.5, 0, 1)
    targets = y[100:].numpy()
    errors = pred != targets
    errors_class_neg1 = np.sum((targets == 0) & errors)
    errors_class_1 = np.sum((targets == 1) & errors)
    accuracy = accuracy_score(pred, targets)
    plot_confusion_matrix(test_set.y.numpy(), pred)
    

    print("predictions: ", predictions)
    print("targets: ", targets)
    print("Accuracy:", accuracy)
    print("Runtime:", training_time * 60, "s")
    print("Errori classe -1: ", errors_class_neg1)
    print("Errori classe 1: ", errors_class_1)
    
if __name__ == "__main__":
    main()
