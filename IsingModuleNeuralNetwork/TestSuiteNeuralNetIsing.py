import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import random

from data_ import SimpleDataset, HiddenNodesInitialization
from NeuralNetIsing import MultiIsingNetwork
from utils import AnnealingSettings

# --- Hyperparameters ---
PARTITION_INPUT = False
NUM_ISING_PERCEPTRONS = 5
SIZE = 10
BATCH_SIZE = 16
EPOCHS = 200
LAMBDA_INIT = -0.01
OFFSET_INIT = 0
LEARNING_RATE_GAMMA = 0.05
LEARNING_RATE_LAMBDA = 0.01
LEARNING_RATE_OFFSET = 0.05
LR_COMBINER = 0.01
CLASSES = [0, 1]
RANDOM_SEED = 42

HN_init = "function"
HN_function = SimpleDataset.offset
HN_fun_args = [-1 / SIZE * 1]

SA_settings = AnnealingSettings()
SA_settings.beta_range = [1, 10]
SA_settings.num_reads = 100
SA_settings.num_sweeps = 1000
SA_settings.sweeps_per_beta = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_csv_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.iloc[:, -1] = df.iloc[:, -1].replace({-1: CLASSES[0], 1: CLASSES[1]})
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y

def plot_training_loss(training_losses, model_name, dataset_name):
    plt.style.use('classic')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(training_losses) + 1), training_losses, color='black', linewidth=2)
    ax.set_xlabel("Epoch", fontsize=14, fontname='serif')
    ax.set_ylabel("Training Loss", fontsize=14, fontname='serif')
    ax.set_title("Training Loss Curve", fontsize=16, fontname='serif', pad=12)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    os.makedirs("Plots", exist_ok=True)
    save_path = f"Plots/{model_name}_{dataset_name}_trainingLoss.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close(fig)

def plot_confusion_matrix_scientific(y_true, y_pred, model_name, dataset_name):
    plt.style.use('classic')
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greys",
        cbar=False, ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
        linewidths=0.5, linecolor='black'
    )
    ax.set_xlabel("Predicted label", fontsize=14, fontname='serif')
    ax.set_ylabel("True label", fontsize=14, fontname='serif')
    ax.set_xticklabels(CLASSES, fontsize=12, fontname='serif')
    ax.set_yticklabels(CLASSES, fontsize=12, fontname='serif', rotation=0)
    ax.set_title("Confusion Matrix", fontsize=16, fontname='serif', pad=12)
    plt.tight_layout()
    os.makedirs("Plots", exist_ok=True)
    save_path = f"Plots/{model_name}_{dataset_name}_confusionMatrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close(fig)

def train_and_eval(X_train, y_train, X_val, y_val, input_dim, dataset_name, run_idx, size):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset = SimpleDataset()
    test_set = SimpleDataset()
    dataset.x = torch.tensor(X_train, dtype=torch.float32)
    dataset.y = torch.tensor(y_train, dtype=torch.float32)
    dataset.data_size = input_dim
    dataset.len = len(y_train)
    test_set.x = torch.tensor(X_val, dtype=torch.float32)
    test_set.y = torch.tensor(y_val, dtype=torch.float32)
    test_set.data_size = input_dim
    test_set.len = len(y_val)

    hn = HiddenNodesInitialization(HN_init)
    hn.function = HN_function
    hn.fun_args = HN_fun_args
    SIZE = size
    if PARTITION_INPUT:
        dataset.resize(SIZE * NUM_ISING_PERCEPTRONS, hn)
        dataset.len = len(dataset.y)
        dataset.data_size = len(dataset.x[0])
        test_set.resize(SIZE * NUM_ISING_PERCEPTRONS, hn)
        test_set.len = len(test_set.y)
        test_set.data_size = len(test_set.x[0])
    else:
        dataset.resize(SIZE, hn)
        dataset.len = len(dataset.y)
        dataset.data_size = len(dataset.x[0])
        test_set.resize(SIZE, hn)
        test_set.len = len(test_set.y)
        test_set.data_size = len(test_set.x[0])

    train_loader = DataLoader(
        TensorDataset(dataset.x, dataset.y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(test_set.x, test_set.y),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = MultiIsingNetwork(
        num_ising_perceptrons=NUM_ISING_PERCEPTRONS,
        sizeAnnealModel=SIZE,
        anneal_settings=SA_settings,
        lambda_init=LAMBDA_INIT,
        offset_init=OFFSET_INIT,
        partition_input=PARTITION_INPUT
    ).to(DEVICE)

    optimizer_grouped_parameters = []
    for p_idx, single_module in enumerate(model.ising_perceptrons_layer):
        optimizer_grouped_parameters.append({'params': [single_module.ising_layer.gamma], 'lr': LEARNING_RATE_GAMMA, 'name': f'gamma_{p_idx}'})
        optimizer_grouped_parameters.append({'params': [single_module.lmd], 'lr': LEARNING_RATE_LAMBDA, 'name': f'lambda_{p_idx}'})
        optimizer_grouped_parameters.append({'params': [single_module.offset], 'lr': LEARNING_RATE_OFFSET, 'name': f'offset_{p_idx}'})
    optimizer_grouped_parameters.append({'params': model.combiner_layer.parameters(), 'lr': LR_COMBINER, 'name': 'combiner'})

    optimizer = torch.optim.SGD(optimizer_grouped_parameters, momentum=0.9)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Training
    training_losses = []
    for epoch in range(EPOCHS):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            model.train()
            optimizer.zero_grad()
            pred = model(x_batch).view(-1)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
        training_losses.append(loss.item())
    
    if(run_idx == 1): 
        # Plot training loss
        plot_training_loss(training_losses, "NuralNetIsing", dataset_name)

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds_tensor = model(test_set.x.to(DEVICE)).cpu().view(-1)
        preds = preds_tensor.numpy()
    predictions = np.where(preds < 0.5, CLASSES[0], CLASSES[1])

    if(run_idx == 1): 
        # Plot confusion matrix
        plot_confusion_matrix_scientific(test_set.y.cpu().numpy(), predictions, "NuralNetIsing", dataset_name)

    return predictions

def main():
    results = []
    for csv_path in glob.glob("datasets/*.csv"):
        X, y = load_csv_dataset(csv_path)
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        input_dim = X.shape[1]
        SIZE = input_dim + 5
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        accs, f1s = [], []
        for run_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            y_pred = train_and_eval(X_train, y_train, X_val, y_val, input_dim, dataset_name, run_idx, SIZE)
            accs.append(accuracy_score(y_val, y_pred))
            f1s.append(f1_score(y_val, y_pred, average='binary'))
        results.append({
            "dataset name": dataset_name,
            "accuracy median": np.median(accs),
            "accuracy mean": np.mean(accs),
            "f1 score median": np.median(f1s),
            "f1 score mean": np.mean(f1s)
        })
        print("Processed dataset:", dataset_name)
    # Create DataFrame and save results
    df_results = pd.DataFrame(results)
    print(df_results)
    df_results.to_csv("NeuralNetIsing_kfold_results.csv", index=False)

if __name__ == "__main__":
    main()