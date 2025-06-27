import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
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
from IsingModule import FullIsingModule
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
SA_settings.num_reads = 1
SA_settings.num_sweeps = 100
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

def prepare_datasets(X_train, y_train, X_val, y_val, input_dim):
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

    SIZE = input_dim + 5 # Adjust size to accommodate input dimension

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
    return dataset, test_set, train_loader, test_loader

def train_and_eval_single(X_train, y_train, X_val, y_val, input_dim):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset, test_set, train_loader, test_loader = prepare_datasets(X_train, y_train, X_val, y_val, input_dim)
    SIZE = input_dim + 5 # Adjust size to accommodate input dimension

    model = FullIsingModule(SIZE, SA_settings, LAMBDA_INIT, OFFSET_INIT).to(DEVICE)
    optimizer = torch.optim.SGD([
        {'params': [model.ising_layer.gamma], 'lr': LEARNING_RATE_GAMMA},
        {'params': [model.lmd], 'lr': LEARNING_RATE_LAMBDA},
        {'params': [model.offset], 'lr': LEARNING_RATE_OFFSET},
    ])
    loss_fn = torch.nn.BCEWithLogitsLoss()

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

    model.eval()
    with torch.no_grad():
        preds_tensor = model(test_set.x.to(DEVICE)).cpu().view(-1)
        preds = preds_tensor.numpy()
    predictions = np.where(preds < 0.5, CLASSES[0], CLASSES[1])
    return predictions

def train_and_eval_net(X_train, y_train, X_val, y_val, input_dim):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset, test_set, train_loader, test_loader = prepare_datasets(X_train, y_train, X_val, y_val, input_dim)
    
    SIZE = input_dim + 5 # Adjust size to accommodate input dimension
    
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

    model.eval()
    with torch.no_grad():
        preds_tensor = model(test_set.x.to(DEVICE)).cpu().view(-1)
        preds = preds_tensor.numpy()
    predictions = np.where(preds < 0.5, CLASSES[0], CLASSES[1])
    return predictions

def main():
    print(f"Using device:{DEVICE}\n")
    results = []
    all_accs = {"Single": {}, "Net": {}}
    for csv_path in glob.glob("datasets/*.csv"):
        X, y = load_csv_dataset(csv_path)
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        input_dim = X.shape[1]
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        accs_single, accs_net = [], []
        for run_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            y_pred_single = train_and_eval_single(X_train, y_train, X_val, y_val, input_dim)
            y_pred_net = train_and_eval_net(X_train, y_train, X_val, y_val, input_dim)
            accs_single.append(accuracy_score(y_val, y_pred_single))
            accs_net.append(accuracy_score(y_val, y_pred_net))
        all_accs["Single"][dataset_name] = accs_single
        all_accs["Net"][dataset_name] = accs_net
        results.append({
            "dataset name": dataset_name,
            "Single median": np.median(accs_single),
            "Single mean": np.mean(accs_single),
            "Net median": np.median(accs_net),
            "Net mean": np.mean(accs_net)
        })
        print(f"Finished dataset: {dataset_name}")

    df_results = pd.DataFrame(results)
    print(df_results)
    df_results.to_csv("ModelComparison_results.csv", index=False)

    # Boxplot scientifico
    plt.style.use('classic')
    fig, ax = plt.subplots(figsize=(10, 6))
    datasets = list(all_accs["Single"].keys())
    positions_single = np.arange(len(datasets)) - 0.15
    positions_net = np.arange(len(datasets)) + 0.15

    bplot1 = ax.boxplot(
        [all_accs["Single"][d] for d in datasets],
        positions=positions_single,
        widths=0.25,
        patch_artist=True,
        boxprops=dict(facecolor='#cccccc'),
        medianprops=dict(color='black', linewidth=2),
        showfliers=False
    )
    bplot2 = ax.boxplot(
        [all_accs["Net"][d] for d in datasets],
        positions=positions_net,
        widths=0.25,
        patch_artist=True,
        boxprops=dict(facecolor='#1f77b4'),
        medianprops=dict(color='black', linewidth=2),
        showfliers=False
    )

    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=12, fontname='serif')
    ax.set_ylabel("Accuracy", fontsize=14, fontname='serif')
    ax.set_title("Model Comparison (5-Fold Cross Validation)", fontsize=16, fontname='serif', pad=12)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend([bplot1["boxes"][0], bplot2["boxes"][0]], ["Single", "Net"], loc="lower right")
    plt.tight_layout()
    plt.savefig("ModelComparison_boxplot.png", dpi=300, bbox_inches='tight', transparent=False)
    plt.close(fig)
    print("All datasets processed. Results and boxplot saved.")

if __name__ == "__main__":
    main()