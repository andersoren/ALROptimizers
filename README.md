# Novel Optimization Method for Training Neural Networks: S-Rprop
This algorithm is based on the mathematical logic of the resilient propagation algorithm suggested by [Reidmiller & Braun](https://ieeexplore.ieee.org/document/298623). Rprop is incompatible with mini-batch learning due to individualised adaptive learning rates being sensitive to local gradient information. Rprop was developed into S-Rprop, separating learning-rate updates from weight updates in the training process. This logic was then extended to SGD with momentum and Adam optimizers. S-Rprop, SGD-Upd, and Adam-Upd have all been coded within the PyTorch framework using the Optimizer class, and are therefore available for further use and testing. The functionality of learning rate mean & standard deviation tracking was implemented for these optimizers in order to better understand the behaviour of the new algorithm. For more information about performance benchmarking and results, see the folder `Academic Research` or read my [thesis paper](http://lup.lub.lu.se/student-papers/record/9166149).

# Requirements
- pip
- Python version >= 3.9
- see requirements file

## Installation

You can install the optimizer directly from this repository. Follow the steps below to clone the repository, install dependencies from `requirements.txt`, and run an example.

### 1. Clone the Repository

```
git clone https://github.com/yourusername/custom-pytorch-optimizer.git
```

### 2. Navigate to repository

```
cd custom-pytorch-optimizer
```

### 3. Install requirements
```
pip install -r requirements.txt
```

### 4. Use optimizers as you please
Edit training loop in `example.py` with custom dataset, architecture and epochs. Then run file in the console:
```
python example.py
```
