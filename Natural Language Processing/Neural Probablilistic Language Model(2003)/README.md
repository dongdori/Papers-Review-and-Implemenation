# Paper Review - Neural Probabilistic Language Model(2003)
## 1. Problem Statement
Statistical Language Model such as n-gram based on conditional probability has problems;  
1. Curse of Dimensionality. It represents word as one hot vector which is length of #Total Vocabs.
2. This model doesn't take in to account Similarity between words.
This paper suggests language model based on Neural Network, which simultaneously learns *Distributed Word Representations* and *parameters to predict Output Word Probability Distribution*.

## 2. Approach
