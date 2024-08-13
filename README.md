# L-Algorithms: Decoupling L and M Mini-Batches for Adaptive Learning-Rates

**Author:** Søren Holst Andersen  
**Date:** June-August 2024

## Introduction

This project explores enhancements to mini-batch training algorithms by implementing and evaluating adaptive learning rates. The focus is on the S-Rprop algorithm, which has shown promising results compared to traditional methods like SGD. Key objectives include:
- Integrating Rprop's adaptive learning-rate scheme with other mini-batch algorithms (SGD and Adam).
- Comparing S-Rprop with advanced algorithms in mini-batch training.
- Expanding tests to larger and more complex datasets, and validating results with separate datasets.

## Implementation

Custom implementations of the SGD and Adam algorithms incorporating the adaptive learning-rate scheme from S-Rprop are available. These modifications are designed to enhance performance and adaptability in various training contexts.

## Hyper-parameter Optimization

Optimization was performed using both Bayesian and grid-search methods to fine-tune hyper-parameters for each algorithm. This process involved exploring various combinations to identify settings that maximize performance.

## Learning-Rate Scheduling

A Cosine Annealing learning rate scheduler was applied to selected algorithms to evaluate its impact on training effectiveness. This scheduling approach adjusts the learning rate dynamically during training to potentially improve convergence.

## Model Validation

Models were validated using the MNIST dataset to assess their performance after training. Observations indicated that certain algorithms, like S-Rprop, experienced slower learning due to larger mini-batch sizes.

## Scaling Up: CIFAR-100

The project expanded to the CIFAR-100 dataset to test the algorithms on a more complex and diverse set of images. Adjustments were made to the model architectures and training parameters to better suit this larger dataset.

## Algorithm Enhancements

Additional features, such as weight decay and momentum, were incorporated into the SGD and S-Rprop algorithms to better evaluate their performance on validation datasets.

## Learning-Rate Tracking

New tracking functionality was added to monitor the mean and standard deviation of the learning rate throughout training. This feature helps in analyzing learning rate behavior and optimizing training processes.

## Future Directions

Future work may involve exploring the effects of Dropout regularization on Rprop training and further optimizing learning-rate parameters for improved model performance.

## Files and Code
- **`Custom Optimizers` Folder:** Contains custom implementations of modified algorithms:
  - **`SGDUpd.py`**: Custom version of the SGD algorithm incorporating adaptive learning rates.
  - **`AdamUpd.py`**: Custom version of the Adam algorithm with adaptive learning rates.
  - **`S-Rprop.py`**: Implementation of the S-Rprop algorithm.

- **`sweep.py`**: Script for performing Bayesian optimization sweeps to find optimal hyper-parameters.

- **`benchmark.py`**: Contains code for grid-search optimization and benchmarking of different algorithms.

- **`CIFAR.py`**: Handles data processing and augmentation for the CIFAR-100 dataset, including image flipping and resizing.

- **`Densetnet.py`**: Implementation of a DenseNet architecture used for reproducing results on CIFAR-100 (though adjustments were made due to dataset size).

- **`resnet9.py`**: Implementation of the ResNet9 architecture, chosen for testing on CIFAR-100 due to its simplicity and effectiveness.

- **`LR_plotting.py`**: Script for plotting the learning rate mean and standard deviation over training epochs, utilizing data exported from wandb.

- **`CIFARsweep.py`**: Contains code for performing grid-search sweeps on various optimizers and hyper-parameters for CIFAR-100.


## Results

Results are presented and commented on in `WhitePaper.md`, as well as detailed information of each experiment.
