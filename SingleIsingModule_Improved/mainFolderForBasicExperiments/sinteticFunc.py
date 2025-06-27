import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from ising_learning_model.data import SimpleDataset, HiddenNodesInitialization, GammaInitialization
from ising_learning_model.sim_anneal_model import SimAnnealModel, AnnealingSettings


def fquad(x):
    return  1.2* (x - 0.5) **  2  - 2

def flin(x):
    return  2*x  - 6


# Model parameters
size = 10
batch_size = 50

learning_rate_gamma = 0.02
learning_rate_lmd = 0.06
learning_rate_offset = 0.1
learning_rate_theta = 0.01
lmd = -0.3
eps = -9.30
training_epochs = 400

ranges_train = [[0,1]]
ranges_test = [[0, 1]]
num_samples_train = 50
num_samples_test = 200

# Simulated Annealing settings
settings_anneal = AnnealingSettings()
settings_anneal.beta_range = [1, 10]  
settings_anneal.num_reads = 100
settings_anneal.num_sweeps = 1000 
settings_anneal.sweeps_per_beta = 1

trainSet = SimpleDataset()
testSet = SimpleDataset()


def main():
    print("\n\n")
    trainSet.create_data_fun(flin, num_samples=num_samples_train, ranges=ranges_train)
    testSet.create_data_fun(flin, num_samples=num_samples_test, ranges=ranges_test)
    
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
    
    # Hidden nodes
    model.settings.hidden_nodes_init = HiddenNodesInitialization("function")
    model.settings.hidden_nodes_init.function = SimpleDataset.offset
    model.settings.hidden_nodes_init.fun_args = [-1/size * 1]
    model.settings.gamma_init = GammaInitialization(mode="zeros")
    
    # Training and testing
    results = model.train(training_set=trainSet, save_params=True, save_samples=True, verbose=False)
    r_test = model.test(testSet)
    predictions = (r_test.results_samples["energy"])

    x_pti = [e[0] for e in testSet.x]
    #model._save_model("C:/Users/Matteo/Documents/GitHub/IsingModel/Ising_Learning_Model/ising-learning-model-main/modelSaved/funcQuad.pkt")
    plotGraphs(x_pti, predictions)
    '''
    model.settings.learning_rate_theta = 1   
    model.settings.learning_rate_gamma = 0
    model.settings.learning_rate_lmd = 0
    model.settings.learning_rate_offset = 0

    results = model.train(training_set=trainSet, save_params=True, save_samples=False, verbose=False)
    r_test = model.test(testSet)
    predictions = (r_test.results_samples["energy"])

    x_pti = [e[0] for e in testSet.x]
    #model._save_model("C:/Users/Matteo/Documents/GitHub/IsingModel/Ising_Learning_Model/ising-learning-model-main/modelSaved/funcQuad.pkt")
    plotGraphs(x_pti, predictions)
    '''
def plotGraphs(x_values, predictions):
    x = np.linspace(0, 1, 200)
    y = flin(x) 
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="target", color='blue')
    plt.scatter(x_values, predictions, color='r', label="predictions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Predictions on flin")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
