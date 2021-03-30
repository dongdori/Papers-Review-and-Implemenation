# Paper Review - Neural Probabilistic Language Model(2003)
## 1. Problem Statement
Statistical Language Model such as n-gram based on conditional probability has such problems;  
1. Curse of Dimensionality. It represents word as one hot vector which is length of total vocabs V, which is very large number.
2. This model doesn't take in to account Similarity between words.
This paper suggests language model based on Neural Network, which simultaneously learns *Distributed Word Representations* and *parameters to predict Output Word Probability Distribution*. 

## 2. Approach
There are 2 goals of Neural Language model.
1. Learn Distributed representation of each words in vocabulary.
2. Learn parameters of model which outputs conditional probability distribution of words.


