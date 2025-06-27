import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils import AnnealingSettings
from data_ import SimpleDataset,HiddenNodesInitialization
from torch.utils.data import TensorDataset, DataLoader
from IsingModule import FullIsingModule
from sklearn.preprocessing import MinMaxScaler

PATH = ""

SIZE = 50
BATCH_SIZE = 1000
EPOCHS = 200

LAMBDA_INIT = -0.03
OFFSET_INIT = -8
LEARNING_RATE_GAMMA = 0.001
LEARNING_RATE_LAMBDA = 0.001
LEARNING_RATE_OFFSET = 0.1

PERCENT_TRAIN = 0.8
PERCENT_TEST = 1 - PERCENT_TRAIN
CLASSES = [0, 1]

DEVICE = "cpu"
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

def main():

    # Load dataset
    data = pd.read_csv(PATH)
    #data.iloc[:, -1] = data.iloc[:, -1].replace({-1: CLASSES[0], 1: CLASSES[1]})

    # Shuffle rows
    data = data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    X = torch.tensor(data.iloc[:, :-1].values.astype(float), dtype=torch.float32)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = torch.tensor(data.iloc[:, -1].values.astype(float), dtype=torch.float32)

    dataset = SimpleDataset()
    test_set = SimpleDataset()

    idx = int(len(data) * PERCENT_TRAIN)
    dataset.x = X[:idx]
    dataset.y = y[:idx]
    dataset.len = len(dataset.y)
    dataset.data_size = len(dataset.x[0])
    
    test_set.x = X[idx:]
    test_set.y = y[idx:]
    test_set.len = len(test_set.y)
    test_set.data_size = len(test_set.x[0])

    #Hidden nodes creation
    hn = HiddenNodesInitialization(HN_init)
    hn.function = HN_function
    hn.fun_args = HN_fun_args

    dataset.resize(SIZE, hn)
    dataset.len = len(dataset.y)
    dataset.data_size = len(dataset.x[0])

    test_set.resize(SIZE, hn)
    test_set.len = len(test_set.y)
    test_set.data_size = len(test_set.x[0])
    
    train_loader = DataLoader(TensorDataset(dataset.x, dataset.y), batch_size=BATCH_SIZE, shuffle=True)

    # 2. Model
    model = FullIsingModule(SIZE, SA_settings, LAMBDA_INIT, OFFSET_INIT).to(DEVICE)
    
    
    optimizer = torch.optim.SGD([
        {'params': [model.ising_layer.gamma], 'lr': LEARNING_RATE_GAMMA},
        {'params': [model.lmd], 'lr': LEARNING_RATE_LAMBDA},
        {'params': [model.offset], 'lr': LEARNING_RATE_OFFSET},
    ], momentum=0.9)  
    loss_fn = nn.MSELoss()
    '''
    optimizer = torch.optim.Adam([
        {'params': [model.ising_layer.gamma], 'lr': 1e-2},
        {'params': [model.lmd], 'lr': 1e-3},
        {'params': [model.offset], 'lr': 1e-1},
    ])  
    loss_fn = nn.BCEWithLogitsLoss()
    '''
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
        preds = model(test_set.x.to(DEVICE)).cpu().numpy()

    predictions = np.where(preds < (abs(CLASSES[1])-abs(CLASSES[0]))/2, CLASSES[0], CLASSES[1]) #class0 must be less than class1
    targets = y[idx:].numpy()
    errors = predictions != targets
    errors_class_neg1 = np.sum((targets == CLASSES[0]) & errors)
    errors_class_1 = np.sum((targets == CLASSES[1]) & errors)
    accuracy = accuracy_score(predictions, targets)
    plot_confusion_matrix(test_set.y.numpy(), predictions)

    print("predictions: ", predictions)
    print("targets: ", targets)
    print("Accuracy:", accuracy)
    print(f"Errori classe {CLASSES[0]}: ", errors_class_neg1)
    print(f"Errori classe {CLASSES[1]}: ", errors_class_1)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[CLASSES[0], CLASSES[1]], yticklabels=[CLASSES[0], CLASSES[1]])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()