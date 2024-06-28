# **L-algorithms: Decoupling L and M mini-batches for adaptive learning-rates**

**Søren Holst Andersen**  
June-August 2024

## Introduction

The results presented in the paper [*Expanding an adaptive learning-rate algorithm to handle mini-batch training*](http://lup.lub.lu.se/student-papers/record/9166149) show that the S-Rprop algorithm competes with SGD on the MNIST dataset with identical architecture and an optimized learning rate, while it outperforms SGD in a deep-learning setting.  
The current goals are to:

- implement Rprop's adaptive learning-rate scheme to other mini-batch algorithms (SGD and Adam) in order to assess if S-Rprop is the best mini-batch algorithm of its kind (or if improved variations exist),
- test S-Rprop against more state-of-the-art algorithms in mini-batch training such as [Adam](https://arxiv.org/abs/1412.6980);
- scale up testing with larger and more complex datasets;
- go beyond training and use a validation dataset to make a better assessment of S-Rprop's results for applications.

## Implementing Adaptive Learning-Rate Scheme for SGD & Adam

The SGD and Adam algorithms, available as open-source code from the [PyTorch website](https://pytorch.org/docs/stable/optim.html), were modified to make use of the adaptive learning-rate scheme explained in [Andersen, 2024].

## Benchmarking New Algorithms

These new algorithms, called *SGD-updated* and *Adam-updated* (names can be improved later), were tested on the MNIST dataset using the CNN architecture from [Andersen, 2024]. The learning rate, $L$ and $M$ hyper-parameters were optimized for each algorithm. 

### Hyper-parameter Optimization

For all algorithms a wandb [sweep](https://docs.wandb.ai/guides/sweeps) was performed with the Bayesian optimization method. Sweeps allow for an intuitive and insightful representation of hyper-parameter relationships. For some algorithms, this revealed clear preferences in terms of hyper-paramter combinations, and this was followed up by a grid-search over a smaller area. For others, Bayesian optimization did not explore the entire parameter space (this method scales poorly) and thus a grid-search was used over a larger area.
During the grid-search, 5 runs were made for each hyper-parameter combination and the loss mean of each point was used to find optimal hyper-parameters giving minimal loss.
In Table 1, the S-Rprop and vanilla SGD results are taken from Andersen (2024). (Explain more in-depth the hyper-parameter process here).

| Algorithm  | Best H-P pair $(L, M, Lr)$ | Min Loss $\mu$ | S.D. $\sigma$ | Min. Loss Epoch |
|----------|-----------------------------|------------------|---------------|------------------|
| S-Rprop    | $30000, 600, 10^{-3}$       | $0.025$          | $0.003$       |                  |
| SGD        | $N/A, 5, 10^{-3}$           | $0.027$          | $0.005$       |                  |
| SGD+M      | $N/A, 25, 10^{-2}$          | $0.013$          | $0.001$       | $5$              |
| SGD-Upd    | $15000, 25, 10^{-1}$        | $0.008$          | $0.001$       | $5$              |
| Adam-Upd   | $12000, 25, 10^{-3}$        | $\mathbf{0.007}$ | $0.001$       | $5$              |
| Adam       | $N/A, 25, 10^{-3}$          | $0.013$          | $0.001$       | $4$              |

*Table 1: Best runs for each optimizer. Mean is presented with standard deviation for 5 runs and the epoch at which minimum loss was reached. 5 epochs of training was used with the MNIST dataset.*

The following are plots of the runs presented in Table 1. On the left is the progression of loss over the entire 5 epochs of training. On the right is a narrowed window showing 4-5 epochs where all algorithms reach their respective values of minimal loss. The S-Rprop and SGD results have been left as they do not reach the same performance level as these algorithms.

<p float="left">
  <img src="./images/Updated%20Algos%20Benchmark%20Speeds.png" title="Image 1" width="500" />
  <img src="./images/Updated%20Algos%20Benchmark.png" title="Image 1" width="500" /> 
</p>

