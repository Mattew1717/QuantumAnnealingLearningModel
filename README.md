# Ising Learning Model

## Overview

This project explores the application of Ising models to machine learning tasks, including classification and regression. The core `ising_learning_model` package is based on and significantly extends the work presented in the "Ising Learning Model", available at [https://github.com/lsschmid/ising-learning-model](https://github.com/lsschmid/ising-learning-model) and detailed in their accompanying research paper (arXiv: https://arxiv.org/pdf/2310.18411).

This repository includes custom implementations of Ising-based Modules, networks of these Nodes, and ensembles, leveraging PyTorch for enhanced flexibility. The project also retains and utilizes different solver backends for the Ising models, including exact solvers, simulated annealing, and Quantum Processing Unit (QPU) based approaches (via D-Wave). The primary goal of this work is to further investigate and expand upon the capabilities of Ising models in machine learning.

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

'''
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn dimod dwave-neal dwave-system networkx PyYAML
'''

## License

Refer to the `LICENSE.md` file for licensing information.
