# Attention is all you need(2017) - Transformer
## 1. Background
Recently, recurrent neural network architectures using RNN cell such as **LSTM, GRU** showed outstanding performance on sequence transduction tasks like **machine translation** or **Abstractive Summarization**.
However, Those RNN approaches are not be able to parallelized during training which results in long training time and high computational cost.

Authors suggest **Transformer**, which is able to be parallelized and solely based on attention mechanism, to solve this problem. I will explain the transformer in context of machine translation.

## 2. Approach
### 2.1. Overview

### 2.2. Self Attention Mechanism
#### 2.2.1. Scaled dot product attention
#### 2.2.2. Multihead Attention

### 2.3. Encoder Structure

### 2.4. Decoder Structure

### 2.5. Training Objective

### 2.6. Advantages of Self Attention
