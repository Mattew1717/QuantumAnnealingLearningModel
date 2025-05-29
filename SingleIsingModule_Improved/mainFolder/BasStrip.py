import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ising_learning_model.data import SimpleDataset, HiddenNodesInitialization, GammaInitialization
from ising_learning_model.sim_anneal_model import SimAnnealModel, AnnealingSettings

# Model parameters
size = 144
cat_factor = 10
batch_size = 80

learning_rate_gamma = 0.005 
learning_rate_lmd = 0
learning_rate_offset = 0.2
lmd = 0.5
eps = 40
training_epochs = 10

num_samples_train = 80
num_samples_test = 80

# Simulated Annealing settings
settings_anneal = AnnealingSettings()
settings_anneal.beta_range = [1, 10]  
settings_anneal.num_reads = 100
settings_anneal.num_sweeps = 1000 
settings_anneal.sweeps_per_beta = 1

trainSet = SimpleDataset()
testSet = SimpleDataset()


def main():
    # Create training and testing datasets
    trainSet.create_data_bas(int(size ** (0.5)), num_samples_train, cat_factor )
    testSet.create_data_bas(int(size ** (0.5)), num_samples_test, cat_factor )

    # Create model
    model = SimAnnealModel(size=size, settings=settings_anneal)
    model.settings.learning_rate = learning_rate_gamma
    model.settings.learning_rate_lmd = learning_rate_lmd
    model.lmd_init_value = lmd
    model.offset_init_value = eps
    model.settings.optim_steps = training_epochs
    model.settings.mini_batch_size = batch_size
    
    # Hidden nodes
    model.settings.hidden_nodes_init = HiddenNodesInitialization("function")
    model.settings.hidden_nodes_init.function = SimpleDataset.offset
    model.settings.hidden_nodes_init.fun_args = [-1/size * 0.8]
    model.settings.gamma_init = GammaInitialization(mode="zeros")
    
    # Training and testing
    results = model.train(training_set=trainSet, save_params=True, save_samples=True, verbose=False, test_set=testSet)
    losses_train = results.results
    losses_test = results.results_test
    results_train = results.results_samples
    results_test = results.results_samples_test
    print(losses_train)
    print(losses_test)
    print(results_train)
    print(results_test) 

    plot_losses(losses_train, losses_test)
    plot_accuracy(results_train, results_test)


def plot_losses(losses_train, losses_test):
    plt.figure(figsize=(10, 5))
    
    # Raggruppiamo per epoca e calcoliamo la media della loss
    train_mean = losses_train.groupby('epoch')['loss'].mean()
    test_mean = losses_test.groupby('epoch')['loss'].mean()

    plt.plot(train_mean.index, train_mean.values, label='Train Loss', marker='o')
    plt.plot(test_mean.index, test_mean.values, label='Test Loss', marker='s')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per epoca')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_accuracy(results_train, results_test):
    # Calcola le predizioni e l'accuratezza per il set di training
    results_train['prediction'] = results_train['energy'].apply(lambda x: 10.0 if x > 5 else 0.0)
    results_train['correct'] = results_train['prediction'] == results_train['target']
    accuracy_train = results_train.groupby('epoch')['correct'].mean()

    # Calcola le predizioni e l'accuratezza per il set di test
    results_test['prediction'] = results_test['energy'].apply(lambda x: 10.0 if x > 5 else 0.0)
    results_test['correct'] = results_test['prediction'] == results_test['target']
    accuracy_test = results_test.groupby('epoch')['correct'].mean()

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(accuracy_train.index, accuracy_train.values, marker='o', linestyle='-', label='Train')
    plt.plot(accuracy_test.index, accuracy_test.values, marker='s', linestyle='--', label='Test')
    plt.title("Accuracy per epoca")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    main()
