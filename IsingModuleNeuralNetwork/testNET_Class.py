import os
import glob
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from utils import AnnealingSettings
from data_ import SimpleDataset, HiddenNodesInitialization
from NeuralNetIsing import MultiIsingNetwork

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
    """
    Returns a unique filename for saving plots.
    """
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

def plot_confusion_matrix_scientific(y_true, y_pred, save_path="confusion_matrix_NET.png"):
    """
    Save a scientific confusion matrix plot.
    """
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

def plot_results_table_scientific(params, accuracy, training_loss, test_loss, errors_class_0, errors_class_1, training_time, save_path="results_table_NET.png"):
    """
    Save a scientific table of results.
    """
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
    """
    Plot training loss curve (loss vs epochs) and save as scientific plot.
    """
    plt.style.use('classic')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(training_losses) + 1), training_losses, marker='o', color='black', linewidth=2)
    ax.set_xlabel("Epoch", fontsize=14, fontname='serif')
    ax.set_ylabel("Training Loss", fontsize=14, fontname='serif')
    ax.set_title("Training Loss Curve", fontsize=16, fontname='serif', pad=12)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    if save_path is None:
        save_path = get_next_plot_filename("training_loss_NET", dataset_name, "png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close(fig)

def load_csv_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.iloc[:, -1] = df.iloc[:, -1].replace({-1: CLASSES[0], 1: CLASSES[1]})
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y

# --- Hyperparameters and Settings ---

PARTITION_INPUT = False
NUM_ISING_PERCEPTRONS = 5
SIZE = 8
BATCH_SIZE = 8
EPOCHS = 200
DATA_INPUT_DIM = 4
TRAINING_SAMPLES = 20
TEST_SAMPLES = 50

LAMBDA_INIT = -0.01
OFFSET_INIT = 0
LEARNING_RATE_GAMMA = 0.02
LEARNING_RATE_LAMBDA = 0.01
LEARNING_RATE_OFFSET = 0.01
LR_COMBINER = 0.001

PERCENT_TRAIN = 0.5
PERCENT_TEST = 1 - PERCENT_TRAIN
CLASSES = [0, 1]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# Simulated Annealing settings
SA_settings = AnnealingSettings()
SA_settings.beta_range = [1, 10]
SA_settings.num_reads = 1
SA_settings.num_sweeps = 1000
SA_settings.sweeps_per_beta = 1

# Hidden nodes parameters
HN_init = "function"
HN_function = SimpleDataset.offset
HN_fun_args = [-1 / SIZE * 1]

# --- Main Execution ---

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
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)

    X, y = generate_xor_balanced(DATA_INPUT_DIM, n_samples_dim=TEST_SAMPLES, shuffle=True)
    X_test = torch.tensor(X, dtype=torch.float32)
    y_test = torch.tensor(y, dtype=torch.float32)

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

    # Model definition
    model = MultiIsingNetwork(
        num_ising_perceptrons=NUM_ISING_PERCEPTRONS,
        sizeAnnealModel=SIZE,
        anneal_settings=SA_settings,
        lambda_init=LAMBDA_INIT,
        offset_init=OFFSET_INIT,
        partition_input=PARTITION_INPUT
    ).to(DEVICE)

    # Optimizer setup
    optimizer_grouped_parameters = []
    for p_idx, single_module in enumerate(model.ising_perceptrons_layer):
        optimizer_grouped_parameters.append({'params': [single_module.ising_layer.gamma], 'lr': LEARNING_RATE_GAMMA, 'name': f'gamma_{p_idx}'})
        optimizer_grouped_parameters.append({'params': [single_module.lmd], 'lr': LEARNING_RATE_LAMBDA, 'name': f'lambda_{p_idx}'})
        optimizer_grouped_parameters.append({'params': [single_module.offset], 'lr': LEARNING_RATE_OFFSET, 'name': f'offset_{p_idx}'})
    optimizer_grouped_parameters.append({'params': model.combiner_layer.parameters(), 'lr': LR_COMBINER, 'name': 'combiner'})

    optimizer = torch.optim.SGD(optimizer_grouped_parameters, momentum=0.9)
    loss_fn = nn.MSELoss()
    #optimizer = torch.optim.Adam(optimizer_grouped_parameters)
    #loss_fn = nn.BCEWithLogitsLoss()

    # Training
    start_time = time.time()
    training_losses = model.train_model(train_loader, optimizer, loss_fn, EPOCHS, DEVICE, print_every=1)
    training_loss = training_losses[-1]
    end_time = time.time()

    # Testing
    predictions_test, targets_test = model.test(test_loader, DEVICE)
    loss_test = loss_fn(torch.tensor(predictions_test, dtype=torch.float32).view(-1), torch.tensor(targets_test, dtype=torch.float32).view(-1)).item()

    predictions = np.where(predictions_test < (abs(CLASSES[1]) - abs(CLASSES[0])) / 2, CLASSES[0], CLASSES[1])
    errors = predictions != targets_test
    errors_class_0 = np.sum((targets_test == CLASSES[0]) & errors)
    errors_class_1 = np.sum((targets_test == CLASSES[1]) & errors)
    print(predictions)
    print(targets_test)
    accuracy = accuracy_score(predictions, targets_test)

    dataset_name = "xor_balanced"

    plot_training_loss(training_losses, dataset_name)
    cm_path = get_next_plot_filename("confusion_matrix_NET", dataset_name, "png")
    table_path = get_next_plot_filename("results_table_NET", dataset_name, "png")
    print(dataset.y)
    plot_confusion_matrix_scientific(targets_test, predictions, save_path=cm_path)

    params = [
        ["Dataset", dataset_name],
        ["Train samples", TRAINING_SAMPLES],
        ["Test samples", TEST_SAMPLES],
        ["Input dim", DATA_INPUT_DIM],
        #["Total samples", len(dataset.x) + len(test_set.x)],
        ["Class labels", CLASSES],
        ["Num Ising Perceptrons", NUM_ISING_PERCEPTRONS],
        ["Size Single Ising Perceptron", SIZE],
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
        ["Partition input", PARTITION_INPUT],
        ["Optimizer", type(optimizer).__name__],
        ["Loss function", type(loss_fn).__name__],
    ]

    plot_results_table_scientific(
        params, accuracy, training_loss, loss_test,
        errors_class_0, errors_class_1, 
        training_time= (end_time - start_time),
        save_path=table_path
    )
