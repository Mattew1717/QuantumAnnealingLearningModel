import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from data_ import SimpleDataset, HiddenNodesInitialization
from IsingModule import FullIsingModule
from NeuralNetIsing import MultiIsingNetwork
from utils import AnnealingSettings
import pandas as pd

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
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

def plot_loss_curve(losses):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses) + 1), losses, color='black', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.close('all')

def plot_regression(x_values, y_true, y_pred):
    x_values = np.array(x_values).flatten()
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    idx_sort = np.argsort(x_values)
    plt.figure(figsize=(8, 6))
    plt.plot(x_values[idx_sort], y_true[idx_sort], label="Target", color='blue')
    plt.scatter(x_values, y_pred, color='red', label="Predictions", alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predictions vs Target")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_scientific(y_true, y_pred):
    plt.style.use('classic')
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greys",
        cbar=False, ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
        linewidths=0.5, linecolor='black'
    )
    ax.set_xlabel("Predicted label", fontsize=14, fontname='serif')
    ax.set_ylabel("True label", fontsize=14, fontname='serif')
    ax.set_title("Confusion Matrix", fontsize=16, fontname='serif', pad=12)
    plt.tight_layout()
    plt.show()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file (YAML)")
    args = parser.parse_args()

    config = load_config(args.config)
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    # 1. Dataset creation
    dataset = SimpleDataset()
    test_set = SimpleDataset()

    if config["task"] == "classification":
        if config.get("xor", False):
            # XOR multidimensionale
            dim = config.get("xor_dim", 2)
            n_train = config.get("xor_train_samples", 1000)
            n_test = config.get("xor_test_samples", 200)
            X_train_np, y_train_np = generate_xor_balanced(dim, n_samples_dim=n_train, shuffle=True, random_seed=config["random_seed"])
            X_test_np, y_test_np = generate_xor_balanced(dim, n_samples_dim=n_test, shuffle=True, random_seed=config["random_seed"]+1)
            X_train = torch.tensor(X_train_np, dtype=torch.float32)
            y_train = torch.tensor(y_train_np, dtype=torch.float32)
            X_test = torch.tensor(X_test_np, dtype=torch.float32)
            y_test = torch.tensor(y_test_np, dtype=torch.float32)
            dataset = SimpleDataset()
            test_set = SimpleDataset()
            dataset.x = X_train
            dataset.y = y_train
            dataset.len = len(y_train)
            dataset.data_size = X_train.shape[1]
            test_set.x = X_test
            test_set.y = y_test
            test_set.len = len(y_test)
            test_set.data_size = X_test.shape[1]
        else:
            # CSV dataset
            PATH = config["dataset_path"]
            PERCENT_TRAIN = config.get("percent_train", 0.8)
            data = pd.read_csv(PATH)
            data = data.sample(frac=1, random_state=config["random_seed"]).reset_index(drop=True)
            X = torch.tensor(data.iloc[:, :-1].values.astype(float), dtype=torch.float32)
            y = torch.tensor(data.iloc[:, -1].values.astype(float), dtype=torch.float32)
            idx = int(len(data) * PERCENT_TRAIN)
            dataset.x = X[:idx]
            dataset.y = y[:idx]
            dataset.len = len(dataset.y)
            dataset.data_size = dataset.x.shape[1]
            test_set.x = X[idx:]
            test_set.y = y[idx:]
            test_set.len = len(test_set.y)
            test_set.data_size = test_set.x.shape[1]

    elif config["task"] == "regression":
        if config["function"] == "quadratic":
            func = lambda x: 1.2 * (x - 0.5) ** 2 - 2
            dataset.create_data_fun(func, num_samples=config["num_samples_train"], ranges=config["ranges_train"])
            test_set.create_data_fun(func, num_samples=config["num_samples_test"], ranges=config["ranges_test"])
        elif config["function"] == "linear":
            func = lambda x: 2 * x - 6
            dataset.create_data_fun(func, num_samples=config["num_samples_train"], ranges=config["ranges_train"])
            test_set.create_data_fun(func, num_samples=config["num_samples_test"], ranges=config["ranges_test"])
        elif config["function"] == "cubic":
            func = lambda x: 4 * (x - 0.5) ** 3 - 2
            dataset.create_data_fun(func, num_samples=config["num_samples_train"], ranges=config["ranges_train"])
            test_set.create_data_fun(func, num_samples=config["num_samples_test"], ranges=config["ranges_test"])
        else:
            raise ValueError(f"Function {config['function']} not supported.")
    else:
        raise ValueError(f"Task {config['task']} not supported.")
    

    # Hidden nodes
    hn = HiddenNodesInitialization(config["hidden_nodes_init"])
    hn.function = getattr(SimpleDataset, config["hidden_nodes_function"])
    hn.fun_args = config["hidden_nodes_args"]

    dataset.resize(config["size"], hn)
    test_set.resize(config["size"], hn)

    # Conversion to tensors
    X_train = dataset.x.detach().clone().float()
    y_train = dataset.y.detach().clone().float().view(-1)
    X_test = test_set.x.detach().clone().float()
    y_test = test_set.y.detach().clone().float().view(-1)

    # 2. Scaler
    if config["scaler"] == "minmax":
        scaler = MinMaxScaler()
    elif config["scaler"] == "standard":
        scaler = StandardScaler()
    else:
        scaler = None

    if scaler is not None:
        X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
        X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config["batch_size"], shuffle=True)

    # 3. SA settings
    sa_settings = AnnealingSettings()
    sa_settings.beta_range = config["sa_settings"]["beta_range"]
    sa_settings.num_reads = config["sa_settings"]["num_reads"]
    sa_settings.num_sweeps = config["sa_settings"]["num_sweeps"]
    sa_settings.sweeps_per_beta = config["sa_settings"]["sweeps_per_beta"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4. Model selection
    if config["model"] == "single":
        model = FullIsingModule(
            config["size"], sa_settings,
            config["lambda_init"], config["offset_init"]
        ).to(device)
        params = [
            {'params': [model.ising_layer.gamma], 'lr': config["learning_rate_gamma"]},
            {'params': [model.lmd], 'lr': config["learning_rate_lambda"]},
            {'params': [model.offset], 'lr': config["learning_rate_offset"]},
        ]
    else:
        model = MultiIsingNetwork(
            num_ising_perceptrons=config["num_ising_perceptrons"],
            sizeAnnealModel=config["size"],
            anneal_settings=sa_settings,
            lambda_init=config["lambda_init"],
            offset_init=config["offset_init"],
        ).to(device)
        params = model.parameters()

    # 5. Optimizer selection
    if config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(params, lr=config.get("learning_rate_gamma", 0.01))
    elif config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(params, lr=config.get("learning_rate_gamma", 0.01))
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported.")

    # 6. Loss function selection
    if config["loss_function"] == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif config["loss_function"] == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError(f"Loss function {config['loss_function']} not supported.")

    # 7. Training
    training_losses = []
    for epoch in range(config["epochs"]):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
        training_losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f}")

    # Plot loss curve
    plot_loss_curve(training_losses)

    # 8. Test
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
        y_true = y_test.numpy()
        if config["task"] == "classification":
            class0, class1 = np.min(y_true), np.max(y_true)
            threshold = (abs(class1) - abs(class0)) / 2
            predictions = np.where(preds < threshold, class0, class1)
            accuracy = np.mean(predictions == y_true)
            print(f"Test Accuracy: {accuracy:.4f}")
            plot_confusion_matrix_scientific(y_true, predictions)
        else:
            mse = np.mean((preds - y_true) ** 2)
            print(f"Test MSE: {mse:.4f}")

            # Plot regression results
            x_plot = [float(e[0]) for e in test_set.x]
            plot_regression(
                x_plot,
                y_true,
                preds,
            )

if __name__ == "__main__":
    main()