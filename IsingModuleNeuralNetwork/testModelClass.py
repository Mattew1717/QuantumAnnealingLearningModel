import os
import glob
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from utils import AnnealingSettings
from data_ import SimpleDataset, HiddenNodesInitialization
from IsingModule import FullIsingModule

# --- Utility Functions ---

def generate_xor_balanced(dim, n_samples_dim=1000, shuffle=True, random_seed=42):
    """
    Generate XOR data in U[0,1]^d with balanced classes.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    samples = np.random.random(size=(2**dim * n_samples_dim, dim))
    for i in range(2**dim):
        signs = np.array([1 if int((i // 2**d) % 2) == 0 else -1 for d in range(dim)])
        samples[i * n_samples_dim:(i + 1) * n_samples_dim] *= signs
    labels = np.sign(np.prod(samples, axis=1))
    if shuffle:
        perm = np.random.permutation(2**dim * n_samples_dim)
        samples = samples[perm]
        labels = labels[perm]
    labels = np.where(labels < 0, 0, 1)
    return samples, labels

def get_next_plot_filename(base_name, dataset_name, ext):
    os.makedirs("Plots", exist_ok=True)
    safe_dataset = os.path.splitext(os.path.basename(dataset_name))[0].replace(" ", "_")
    pattern = f"Plots/{base_name}_{safe_dataset}_*.{ext}"
    existing = glob.glob(pattern)
    nums = []
    for f in existing:
        try:
            num = int(os.path.splitext(f)[0].split("_")[-1])
            nums.append(num)
        except ValueError:
            continue
    next_num = max(nums) + 1 if nums else 1
    return f"Plots/{base_name}_{safe_dataset}_{next_num}.{ext}"

def plot_confusion_matrix_scientific(y_true, y_pred, save_path="confusion_matrix_Model.png"):
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
    ax.set_xticklabels([CLASSES[0], CLASSES[1]], fontsize=12, fontname='serif')
    ax.set_yticklabels([CLASSES[0], CLASSES[1]], fontsize=12, fontname='serif', rotation=0)
    ax.set_title("Confusion Matrix", fontsize=16, fontname='serif', pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close(fig)

def plot_results_table_scientific(params, accuracy, training_loss, test_loss, errors_class_0, errors_class_1, training_time, save_path="results_table_Model.png"):
    plt.style.use('classic')
    fig, ax = plt.subplots(figsize=(8, len(params) * 0.35 + 2))
    ax.axis('off')
    table_data = params + [
        ["Accuracy", f"{accuracy:.4f}"],
        ["Training Loss", f"{training_loss:.4f}"],
        ["Test Loss", f"{test_loss:.4f}"],
        [f"Errors class {CLASSES[0]}", errors_class_0],
        [f"Errors class {CLASSES[1]}", errors_class_1],
        ["Training time (s)", f"{training_time:.2f}"]
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=["Parameter", "Value"],
        cellLoc='left',
        loc='center',
        edges='horizontal'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.3)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        text = cell.get_text()
        text.set_fontname('serif')
        if row == 0:
            cell.set_facecolor('#f5f5f5')
            text.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close(fig)

def plot_training_loss(training_losses, dataset_name, save_path=None):
    plt.style.use('classic')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(training_losses) + 1), training_losses, marker='o', color='black', linewidth=2)
    ax.set_xlabel("Epoch", fontsize=14, fontname='serif')
    ax.set_ylabel("Training Loss", fontsize=14, fontname='serif')
    ax.set_title("Training Loss Curve", fontsize=16, fontname='serif', pad=12)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    if save_path is None:
        save_path = get_next_plot_filename("training_loss_Model", dataset_name, "png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close(fig)

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.iloc[:, -1] = df.iloc[:, -1].replace({-1: CLASSES[0], 1: CLASSES[1]})
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y

# --- Hyperparameters and Settings ---

SIZE = 8
BATCH_SIZE = 16
EPOCHS = 200

LAMBDA_INIT = -0.01
OFFSET_INIT = 0
LEARNING_RATE_GAMMA = 0.05
LEARNING_RATE_LAMBDA = 0.05
LEARNING_RATE_OFFSET = 0.05

PERCENT_TRAIN = 0.5 
PERCENT_TEST = 1 - PERCENT_TRAIN
CLASSES = [0, 1]

DATA_INPUT_DIM = 3
TRAINING_SAMPLES = 200
TEST_SAMPLES = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# Simulated Annealing settings
SA_settings = AnnealingSettings()
SA_settings.beta_range = [1, 10]
SA_settings.num_reads = 1
SA_settings.num_sweeps = 100
SA_settings.sweeps_per_beta = 1

# Hidden nodes parameters
HN_init = "function"
HN_function = SimpleDataset.offset
HN_fun_args = [-0.1]

if __name__ == '__main__':
    torch.set_num_threads(os.cpu_count())
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --- Data preparation (XOR 2D) ---
    X, y = generate_xor_balanced(DATA_INPUT_DIM, n_samples_dim=TRAINING_SAMPLES, shuffle=True)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    idx = torch.randperm(len(y))
    X, y = X[idx], y[idx]
    split = int(len(y) * PERCENT_TRAIN)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    dataset = SimpleDataset()
    test_set = SimpleDataset()
    dataset.x = X_train
    dataset.y = y_train
    dataset.data_size = X_train.shape[1]
    dataset.len = len(y_train)
    test_set.x = X_test
    test_set.y = y_test
    test_set.data_size = X_test.shape[1]
    test_set.len = len(y_test)

    # Hidden nodes initialization
    hn = HiddenNodesInitialization(HN_init)
    hn.function = HN_function
    hn.fun_args = HN_fun_args

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

    # Model definition
    model = FullIsingModule(SIZE, SA_settings, LAMBDA_INIT, OFFSET_INIT).to(DEVICE)
    
    optimizer = torch.optim.SGD([
        {'params': [model.ising_layer.gamma], 'lr': LEARNING_RATE_GAMMA},
        {'params': [model.lmd], 'lr': LEARNING_RATE_LAMBDA},
        {'params': [model.offset], 'lr': LEARNING_RATE_OFFSET},
    ], momentum=0.9)
    loss_fn = nn.MSELoss()
    '''
    optimizer = torch.optim.Adam([
        {'params': [model.ising_layer.gamma], 'lr': LEARNING_RATE_GAMMA},
        {'params': [model.lmd], 'lr': LEARNING_RATE_LAMBDA},
        {'params': [model.offset], 'lr': LEARNING_RATE_OFFSET},
    ])
    loss_fn = nn.BCEWithLogitsLoss()
    '''
    # Training
    start_time = time.time()
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
    training_loss = training_losses[-1]
    end_time = time.time()

    # Plot training loss
    dataset_name = "xor_balanced"
    plot_training_loss(training_losses, dataset_name)

    # Test
    model.eval()
    with torch.no_grad():
        preds_tensor = model(dataset.x.to(DEVICE)).cpu().view(-1)
        loss_test = loss_fn(preds_tensor, dataset.y.cpu().view(-1)).item()
        preds = preds_tensor.numpy()

    predictions = np.where(preds < 0.5, CLASSES[0], CLASSES[1])
    targets = dataset.y.cpu().numpy()
    errors = predictions != targets
    errors_class_0 = np.sum((targets == CLASSES[0]) & errors)
    errors_class_1 = np.sum((targets == CLASSES[1]) & errors)
    accuracy = accuracy_score(predictions, targets)

    # Save scientific plots
    cm_path = get_next_plot_filename("confusion_matrix_Model", dataset_name, "png")
    table_path = get_next_plot_filename("results_table_Model", dataset_name, "png")

    plot_confusion_matrix_scientific(targets, predictions, save_path=cm_path)

    params = [
        ["Dataset", dataset_name],
        ["Train samples", TRAINING_SAMPLES],
        ["Test samples", TEST_SAMPLES],
        ["Input dim", DATA_INPUT_DIM],
        ["Class labels", CLASSES],
        ["Model size", SIZE],
        ["Batch size", BATCH_SIZE],
        ["Epochs", EPOCHS],
        ["Lambda init", LAMBDA_INIT],
        ["Offset init", OFFSET_INIT],
        ["Learning rate gamma", LEARNING_RATE_GAMMA],
        ["Learning rate lambda", LEARNING_RATE_LAMBDA],
        ["Learning rate offset", LEARNING_RATE_OFFSET],
        ["Hidden nodes init", HN_init],
        ["Hidden nodes function", HN_function.__name__ if callable(HN_function) else HN_function],
        ["Hidden nodes args", f"{HN_fun_args[0]:.4f}"],
        ["SA Beta range", SA_settings.beta_range],
        ["SA num reads", SA_settings.num_reads],
        ["SA num sweeps", SA_settings.num_sweeps],
        ["SA sweeps per beta", SA_settings.sweeps_per_beta],
        ["Optimizer", type(optimizer).__name__],
        ["Loss function", type(loss_fn).__name__],
    ]

    plot_results_table_scientific(
        params, accuracy, training_loss, loss_test,
        errors_class_0, errors_class_1,
        training_time=(end_time - start_time),
        save_path=table_path
    )