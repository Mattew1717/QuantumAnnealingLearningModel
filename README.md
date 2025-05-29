# Ising Model Machine Learning Project

## Overview

This project explores the application of Ising models to various machine learning tasks, including classification and regression. It features implementations of single Ising-based perceptrons, networks of these perceptrons, and ensembles. The project also includes different solver backends for the Ising models, such as exact solvers, simulated annealing, and Quantum Processing Unit (QPU) based approaches (via D-Wave).

## Directory Structure
```
.
├── LICENSE.md
├── README.md
├── datasets/
│ ├── 01_iris_setosa_versicolor.csv
│ ├── 02_iris_setosa_virginica.csv
│ ├── 03_iris_versicolor_virginica.csv
│ ├── 04_seeds_1_2.csv
│ └── breast-cancer.csv
├── IsingModuleNeuralNetwork/
│ ├── data_.py # Custom data handling for the neural network module
│ ├── multiPerceptronEnsemble.py # Implementation of an ensemble of multi-perceptrons
│ ├── perceptronModule.py # Module for an Ising-based perceptron
│ ├── perceptronNet.py # Network of Ising-based perceptrons
│ ├── testNET_Class.py # Test script for network-based classification
│ ├── testNET_Regr.py # Test script for network-based regression
│ ├── testPercClass.py # Test script for single perceptron classification
│ ├── testPercRegr.py # Test script for single perceptron regression
│ └── utils.py # Utility functions for the IsingModuleNeuralNetwork
├── SingleIsingModule_Improved/
│ └── ising_learning_model/ # Core package for Ising model learning
│ ├── init.py # Initializes the ising_learning_model package
│ ├── data.py # Data handling and dataset classes for ising_learning_model
│ ├── exact_model.py # Exact solver for the Ising model
│ ├── model.py # Base model class for Ising learning
│ ├── qpu_model.py # D-Wave QPU based Ising model solver
│ ├── sim_anneal_model.py # Simulated Annealing based Ising model solver
│ └── utils.py # Utility functions for ising_learning_model
└── mainFolder/
├── BasStrip.py # Main script for the Bars and Stripes problem
├── breastCancer.py # Main script for the breast cancer dataset classification
├── IRISmain.py # Main script for the Iris dataset classification
├── MNISTmain.py # Main script for the MNIST dataset classification
├── sinteticFunc.py # Main script for synthetic function approximation (regression)
└── XORmain.py # Main script for the XOR problem classification
```
## Key Components

*   **`datasets/`**: Contains various CSV files used for training and testing the models. These include datasets for problems like Iris classification, seeds classification, and breast cancer diagnosis.

*   **`IsingModuleNeuralNetwork/`**: This directory houses a custom-built framework for creating neural network-like structures using Ising-based perceptrons.
    *   It includes modules for single perceptrons (`perceptronModule.py`), networks of perceptrons (`perceptronNet.py`), and ensembles (`multiPerceptronEnsemble.py`).
    *   Dedicated scripts for testing classification and regression capabilities of both single perceptrons and networks are provided.

*   **`SingleIsingModule_Improved/ising_learning_model/`**: This is the core library providing the foundational elements for Ising model-based learning.
    *   It defines the base `model.py` class and specific implementations for different solvers: `exact_model.py`, `sim_anneal_model.py` (Simulated Annealing), and `qpu_model.py` (Quantum Processing Unit).
    *   Custom data handling (`data.py`) and utility functions (`utils.py`) specific to this library are included.

*   **`mainFolder/`**: Contains the main executable Python scripts that demonstrate the application of the `ising_learning_model` components to various machine learning problems. Each script is typically tailored to a specific dataset or problem type (e.g., XOR, MNIST, Breast Cancer, Synthetic Functions, Bars and Stripes).

## Problem Types Addressed

The project tackles several machine learning problems:

*   **Classification**:
    *   XOR (`XORmain.py`, `multiPerceptronEnsemble.py`)
    *   Breast Cancer (`breastCancer.py`, `testPercClass.py`, `testNET_Class.py`)
    *   Iris (`IRISmain.py`)
    *   MNIST (`MNISTmain.py`)
    *   Bars and Stripes (`BasStrip.py`, `bas_data.ipynb`)
*   **Regression**:
    *   Synthetic Function Approximation (`sinteticFunc.py`, `testPercRegr.py`, `testNET_Regr.py`, `function_data.ipynb`)

## How to Run

1.  Ensure all dependencies are installed (see below).
2.  Navigate to the `mainFolder/` directory to run the example scripts.
3.  Execute a specific script using Python, for example:
    ```bash
    python XORmain.py
    ```
4.  Scripts within `IsingModuleNeuralNetwork/` (e.g., `testNET_Class.py`) can also be run directly to test those specific components.

## Dependencies

This project likely requires common Python libraries for scientific computing and machine learning, such as:
*   Python 3.x
*   PyTorch
*   NumPy
*   Pandas
*   Matplotlib
*   Seaborn
*   scikit-learn
*   D-Wave Ocean SDK (for QPU and simulated annealing components, e.g., `dimod`, `neal`, `dwave-system`)

Please ensure these are installed in your Python environment.

## License

Refer to the `LICENSE.md` file for licensing information.
