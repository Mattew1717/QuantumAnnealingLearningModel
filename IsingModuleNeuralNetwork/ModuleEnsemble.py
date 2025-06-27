import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SingleIsingModule_Improved.ising_learning_model.data import SimpleDataset, HiddenNodesInitialization, GammaInitialization
from SingleIsingModule_Improved.ising_learning_model.sim_anneal_model import SimAnnealModel, AnnealingSettings
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.nn as nn

#MODEL PARAMETERS
NUMBER_OF_MODELS = 4
SINGLE_MODEL_SIZE = 30
BATCH_SIZE = 10000
LEARNING_RATE_GAMMA = 0.25
LEARNING_RATE_LAMBDA = 0.001
LEARNING_RATE_OFFSET = 1
LEARNING_RATE_THETA = 1
LAMBDA_INIT = -0.03
OFFSET_INIT = -8
OFFSET_FUN_ARGS = [-1/SINGLE_MODEL_SIZE * 1]
TRAINING_EPOCHS = 50

GAMMA_INITIALIZATION = "zeros"
HIDDEN_NODES_INITIAL = "function"

CLASSES = [0, 1]
RANDOM_SEED = 42

#SIMULATED ANNEALING SETTINGS
BETA_RANGE = [1, 10]
NUM_READS = 1 
NUM_SWEEPS = 1000
SWEEPS_PER_BETA = 1

#FINAL PERCEPTRON SETTINGS
LEARNING_RATE_FP = 0.01
EPOCHS_FP = 100

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

class FinalPerceptron(nn.Module):
    def __init__(self, input_size=NUMBER_OF_MODELS):
        super(FinalPerceptron, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

class EnsembleNet:

    def __init__(self, input_size=NUMBER_OF_MODELS):

         # Model parameters
        size = SINGLE_MODEL_SIZE

        # Simulated Annealing settings
        settings_anneal = AnnealingSettings()
        settings_anneal.beta_range = BETA_RANGE
        settings_anneal.num_reads = NUM_READS
        settings_anneal.num_sweeps = NUM_SWEEPS
        settings_anneal.sweeps_per_beta = SWEEPS_PER_BETA

        self.hidden_models = [SimAnnealModel(size=size, settings=settings_anneal) for _ in range(NUMBER_OF_MODELS)]
        for i, model in enumerate(self.hidden_models):
            
            model.settings.learning_rate_gamma = LEARNING_RATE_GAMMA
            model.settings.learning_rate_lmd = LEARNING_RATE_LAMBDA
            model.settings.learning_rate_offset = LEARNING_RATE_OFFSET
            model.settings.learning_rate_theta = LEARNING_RATE_THETA
            model.lmd_init_value = LAMBDA_INIT
            model.offset_init_value = OFFSET_INIT
            model.settings.optim_steps = TRAINING_EPOCHS
            model.settings.mini_batch_size = BATCH_SIZE
            model.settings.hidden_nodes_init = HiddenNodesInitialization(HIDDEN_NODES_INITIAL)
            model.settings.hidden_nodes_init.function = SimpleDataset.offset
            model.settings.hidden_nodes_init.fun_args = [OFFSET_FUN_ARGS]
            model.settings.gamma_init = GammaInitialization(GAMMA_INITIALIZATION)

        self.output_layer = FinalPerceptron()
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.output_layer.parameters(), lr=LEARNING_RATE_FP)

    def train_hidden(self, dataset):
        for i, model in enumerate(self.hidden_models):
            start_feature = i * SINGLE_MODEL_SIZE
            end_feature = (i + 1) * SINGLE_MODEL_SIZE

            # Crea sotto-dataset
            sub_dataset = SimpleDataset()
            sub_dataset.x = dataset.x[:, start_feature:end_feature]
            sub_dataset.y = dataset.y
            sub_dataset.len = dataset.len
            sub_dataset.data_size = len(sub_dataset.x[0]) 

            model.train(training_set=sub_dataset, save_params=False, save_samples=False, verbose=False)

    def forward_hidden(self, test_set):
        features = []
        for i, model in enumerate(self.hidden_models):
            start_feature = i * SINGLE_MODEL_SIZE
            end_feature = (i + 1) * SINGLE_MODEL_SIZE

            sub_test = SimpleDataset()
            sub_test.x = test_set.x[:, start_feature:end_feature]
            sub_test.y = test_set.y
            sub_test.len = test_set.len
            sub_test.data_size = len(sub_test.x[0]) 

            result = model.test(sub_test)
            pred = result.results_samples["energy"].values
            features.append(torch.tensor(pred).unsqueeze(1))
        return torch.cat(features, dim=1).float()

    def train_output_layer(self, features, labels, epochs=EPOCHS_FP):
        for _ in range(epochs):
            outputs = self.output_layer(features)
            loss = self.criterion(outputs.squeeze(), labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

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


    #Hidden nodes creation
    hn = HiddenNodesInitialization("function")
    hn.function = SimpleDataset.offset
    hn.fun_args = [-1/(SINGLE_MODEL_SIZE*NUMBER_OF_MODELS) * 1]
    dataset.resize((SINGLE_MODEL_SIZE*NUMBER_OF_MODELS), hn)
    dataset.len = len(dataset.y)
    dataset.data_size = len(dataset.x[0])
    
    #Hidden nodes creation
    hn = HiddenNodesInitialization("function")
    hn.function = SimpleDataset.offset
    hn.fun_args = [-1/(SINGLE_MODEL_SIZE*NUMBER_OF_MODELS) * 1]
    test_set.resize((SINGLE_MODEL_SIZE*NUMBER_OF_MODELS), hn)
    test_set.len = len(test_set.y)
    test_set.data_size = len(test_set.x[0])

    # Create models net
    model = EnsembleNet(input_size=NUMBER_OF_MODELS)

    # Training single models
    model.train_hidden(dataset)

    # generate results after training
    res_train = model.forward_hidden(dataset)
    res_test = model.forward_hidden(test_set)

    # Training final perceptron
    model.train_output_layer(res_train, dataset.y, epochs=EPOCHS_FP)

    # final testing
    outputs = model.output_layer(res_test).detach().numpy().squeeze()
    pred = np.where(outputs < 0.5, 0, 1)
    accuracy = accuracy_score(pred, test_set.y.numpy())
    print("final accuracy:", accuracy)
    plot_confusion_matrix(test_set.y.numpy(), pred)


    
if __name__ == "__main__":
    main()
