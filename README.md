# Transformer_from_scratch

I built this project in order to exactly understand the architecture of a basic Transformer. I also wanted to play with the positional encoding, for which I wanted to have full control of my own libraries.

In the past I've already completed a project on building a Neural Network from scratch, also implementing its training stage by building the automatic differentiation framework from scratch, without using Pytorch.
For Transformers the architecure is far more complex than for a regular Multilayer Perceptron and optimizing a optimizer would require much more repetitive work that would not increase my understanding or the efficiency of already implemented PyTorch modules. Therefore by 'Transformer from Scratch', it refers to its architecture: the Multi-Headed Attention, the Normalization Layers, the Feed Forward Layers, the Embedder and even a rudimentary tokenizer. In the same way, the Embedder is closer to more basic tools from Neural Networks so I just implemented the corresponding ones from Pytorch modules.
