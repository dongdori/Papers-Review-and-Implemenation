# Attention is all you need(2017) - Transformer
## 1. Background
Recently, recurrent neural network architectures using RNN cell such as **LSTM, GRU** showed outstanding performance on sequence transduction tasks like **machine translation** or **Abstractive Summarization**.
However, Those RNN approaches are not be able to parallelized during training which results in long training time and high computational cost.

Authors suggest **Transformer**, which is able to be parallelized and solely based on attention mechanism, to solve this problem. I will explain the transformer in context of machine translation.

## 2. Approach
### 2.1. Overview
<img width="260" alt="20210610_001430" src="https://user-images.githubusercontent.com/70640345/121381752-f34f7600-c980-11eb-847a-dfad6a406195.png">

Picture above shows a single layer of Encoder and Decoder. Transformer consists of total 6 layers. Single encoder layer consists of 2 sublayers(**Multihead Self Attention layer & Positionwise feed forward layer**) and corresponding single decoder layer consists of 3 sublayers(**Masked Multihead Attention layer, Multihead Attention layer, Positionwise feed forward layer**).

During training phase, source sentence and target sentence are encoded into *seq_len x d<sub>model</sub>* dimensional matrix, which is computed as a sum of token embeddings and positional encoding. 

### 2.2. Multihead Self Attention Mechanism
Basic attention mechanism enables decoder to attend to specific part of source sentence. However, self attention mechanism enables encoder and decoder to consider context of sentence itself. Multihead self attention mechanism incorporates several steps.

#### 2.2.1. Scaled dot product Self Attention
The goal of scaled dot product self attention is to compute relationship between each words in a single sentence. 

Single Input setence can be represented as *max_len x d<sub>model</sub>* dimensional matrix, in which *d<sub>model</sub>* is dimension of embedding vector.
Let's denote input sentence embedding matrix as I. 

I is diverges into Q(query), K(keys), V(values) which are *max_len x d<sub>model</sub>* dimensional and all identical. 
Q and K are used for computing attention vector *a<sub>w</sub>* for each words. And contextual representation of each words are computed as weighted average of V.

Formulation of Scaled dot product attention is as below.

<img width="181" alt="20210611_000618" src="https://user-images.githubusercontent.com/70640345/121549583-f44bda80-ca48-11eb-9263-9dbea6ecb0f7.png">

A<sub>i,j</sub> means attention score between word i and word j, where A = *Softmax(...)* term. and each rows of A*V* means contextual representation of each words. 

#### 2.2.2. Multihead Attention

### 2.3. Encoder Structure

### 2.4. Decoder Structure

### 2.5. Training Objective

### 2.6. Advantages of Self Attention
