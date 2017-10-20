# Code release for "Bayesian Compression for Deep Learning"


In "Bayesian Compression for Deep Learning" we take an information theoretic take on the compression of neural networks. We explicitly revisit the connection between the minimum description length principle and variational inference.

Compression of neural networks with up to 700x and speed up up to 50x (CPU to sparse GPU).

We achive these results by learning additive noise to the weights. We visualize the learning process in the following figures examplarily for a dense network with 300 and 100 connections. Whitness represents redundacy, red and blue positive and negative weights respectively.

|First layer weights |Second Layer weights|
| :------ |:------: |
|![alt text](./figures/weight0_e.gif "First layer weights")|![alt text](./figures/weight1_e.gif "Second Layer weights")|

For dense networks it is also simple to reconstruct input feature impartance. We show this for a mask and 5 randomly chosen digits.
![alt text](./figures/pixel.gif "Pixel importance")


## Results


| Model             | Method | Error [%] | Compression <br/>after pruning | Compression after <br/> precision reduction |
| ------            | :------ |:------: | ------: |------: |
|LeNet-5-Caffe      |[DC](https://arxiv.org/abs/1510.00149)   |   0.7 |    6*    |  -|
|                   |[DNS](https://arxiv.org/abs/1608.04493)    |   0.9 |    55*    |  -|
|                   |[SWS](https://arxiv.org/abs/1702.04008)     |   1.0 |    100*    |  -|
|                   |[Sparse VD](https://arxiv.org/pdf/1701.05369.pdf) |   1.0 |    63*    |  228|
|                   |BC-GNJ  |   1.0 |    108*    |  361|
|                   |BC-GHS  |   1.0 |    156*    |  419|
|  VGG              |BC-GNJ  |   8.6 |    14*    |  56|
|                   |BC-GHS  |   9.0 |    18*    |  59|

## Usage
We provide an implementation in pyTorch for linear and convolutional layers for the group normal-Jeffreys prior (aka Group Variational Dropout) via
```python
import BayesianLayer
```
Layers can be inclued equivalently to their frequentist counter parts.
```python
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # activation
            self.relu = nn.ReLU()
            # layers
            self.fc1 = BayesianLayer.LinearGroupVD(28 * 28, 300, clip_var=0.04)
            self.fc2 = BayesianLayer.LinearGroupVD(300, 100)
            self.fc3 = BayesianLayer.LinearGroupVD(100, 10)
            # layers including kl_divergence
            self.kl_list = [self.fc1, self.fc2, self.fc3]

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)

        def kl_divergence(self):
            KLD = 0
            for layer in self.kl_list:
                KLD += layer.kl_divergence()
            return KLD
```
The only addional effort is to define a KL-divergence. Which is of need for the optimisation of the variational lower bound
```python
N = 60000.
discrimination_loss = nn.functional.cross_entropy

def objective(output, target, kl_divergence):
    discrimination_error = discrimination_loss(output, target)
    return discrimination_error + kl_divergence / N
```
## Run an example
We provide a simple example, the LeNet-300-100 trained with normal-Jeffreys prior.
```sh
python example.py
```

## Retraining a regular neural network
Often times we wish to compress an already existing network. To retrain a pretrained network just inialize the weights when creating an equivalent Bayesian network
```python
    BayesianLayer.LinearGroupVD(28*28, 300,init_weight=pretrained_weight, init_bias=pretrained_bias)
```
## *Reference*
The paper "Bayesian Compression for Deep Learning" has been accepted to NIPS 2017. Please cite us:

    @article{louizos2017bayesian,
      title={Bayesian Compression for Deep Learning},
      author={Louizos, Christos and Ullrich, Karen and Welling, Max},
      journal={Conference on Neural Information Processing Systems (NIPS)},
      year={2017}
    }