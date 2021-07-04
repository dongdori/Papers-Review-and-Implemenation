# Attention is all you need(2017) - Transformer
## 1. Background
  Recently, recurrent neural network architectures using RNN cell such as **LSTM, GRU** showed outstanding performance on sequence transduction tasks like **machine translation** or **Abstractive Summarization**.
  However, Those RNN approaches are not be able to parallelized during training which results in long training time and high computational cost.

  Authors suggest ```Transformer```, which is able to be parallelized and solely based on attention mechanism, to solve this problem. I will explain the transformer in context of machine translation.

## 2. Approach
### 2.1. Overview
<img width="260" alt="20210610_001430" src="https://user-images.githubusercontent.com/70640345/121381752-f34f7600-c980-11eb-847a-dfad6a406195.png">

  Picture above shows a single layer of Encoder and Decoder. Transformer consists of total 6 layers. Single encoder layer consists of 2 sublayers(```MultiheadSelfAttention``` layer & ```PositionwiseFeedForward``` layer) and corresponding single decoder layer consists of 3 sublayers(```Masked Multihead Attention``` layer, ```Multihead Attention layer```, ```Positionwise feed forward``` layer).

  During training phase, source sentence and target sentence are encoded into ```seq_len``` x ```d<sub>model</sub>``` dimensional matrix, which is computed as a sum of token embeddings and positional encoding. 

### 2.2. Multihead Self Attention Mechanism
Basic attention mechanism enables decoder to attend to specific part of source sentence. However, self attention mechanism enables encoder and decoder to consider context of sentence itself. Multihead self attention mechanism incorporates several steps.

#### 2.2.1. Scaled dot product Self Attention
The goal of scaled dot product self attention is to compute relationship between each words in a single sentence. 

Single Input setence can be represented as ```max_len``` x ```d<sub>model</sub>``` dimensional matrix, in which ```d<sub>model</sub>``` is dimension of embedding vector.
Let's denote input sentence embedding matrix as I. 

I is diverges into ```Q```(query), ```K```(keys), ```V```(values) which are ```max_len``` x ```d<sub>model</sub>``` dimensional and all identical. 
```Q``` and ```K``` are used for computing attention vector ```a<sub>w</sub>``` for each words. And contextual representation of each words are computed as weighted average of ```V```.

Formulation of Scaled dot product attention is as below.

<img width="181" alt="20210611_000618" src="https://user-images.githubusercontent.com/70640345/121549583-f44bda80-ca48-11eb-9263-9dbea6ecb0f7.png">

A<sub>i,j</sub> means attention score between word i and word j, where A = *Softmax(...)* term.
And each rows of ```tf.matmul(A, V)``` is weighted average of each rows in ```V```, where each weights are each row of ```A```. Output of Scaled dot product attention is ```len``` x ```d<sub>model</sub>``` dimensional.

#### 2.2.2. Multihead Attention
Let ```h``` denotes the number of heads.
```Q```, ```K```, ```V``` are linearly projected into ```len``` x ```d<sub>h</sub>``` matrix through ```h``` heads. ```d<sub>h</sub>``` = ```d<sub>model</sub>``` / ```h```.

Let Q<sub>i</sub>, K<sub>i</sub>, V<sub>i</sub> denotes projection of Q, K, V in i-th head. 

And let head<sub>i</sub> = Attention(Q<sub>i</sub>, K<sub>i</sub>, V<sub>i</sub>).

And concatenated matrix <head<sub>1</sub> ... head<sub>h</sub>>, which is <i>len x h*d<sub>h</sub></i>, is linearly transformed into *len x d<sub>model</sub>* dimensional matrix

Below is brief illustration of Multihead Attention.

<img width="333" alt="20210612_000450" src="https://user-images.githubusercontent.com/70640345/121707774-e1e9a380-cb11-11eb-9497-5b68884c9fe4.png">




### 2.3. Encoder Structure
Encoder consists of 2 sublayers, which are **Multihead Attention layer** and **Positionwise Feed Forward layer**. And at the top of each sublayers, there are **residual connection** and **normalization layer**.

#### 2.3.1. Multihead Attention layer
Input embedding *X<sub>input</sub>*, which is a sum of token embedding and positional encoding, is *max_len* x *d<sub>model</sub>* dimensional matrix.
Multihead Attention layer outputs *max_len* x *d<sub>model</sub>* dimensional matrix, where each rows are **contextual representations of words**.

Final output *X<sub>output</sub>* is computed as *LayerNorm(X<sub>input</sub> + MultiheadAttention(X<sub>input</sub>))*.

#### 2.3.2. Positionwise Feed Forward layer
'Position' means each rows of X<sub>output</sub>, which denotes contextual word representation. In positionwise feed forward layer, all rows are linearly transformed with same weights. It can be also expressed as 1D-convolution operation.  

Formula is, 

*FFN(x) = ReLu(xW<sub>1</sub> + b<sub>1</sub>)W<sub>2</sub> + b<sub>2</sub>*

where *W<sub>1</sub>* is *len* x *2048* dimensional, and *W<sub>2</sub>* is *len* x *d<sub>model</sub>* dimensional.
Each weight matrices are composed of identical columns(to perform **position-wise linear transformation**).  

### 2.4. Decoder Structure
Decoder consists of 3 sublayers, which are **Masked multihead attention layer, Multihead attention layer, Positionwise feed forward layer**. Each sublayers of decoder are similar to that of encoder, except **Masked Multihead attention layer**.

As objective of decoder at inference phase is to generate translation, decoder have to be auto-regressive, which leads to **Masked Multihead Attention layer**.    
#### 2.4.1. Masked Multihead Attention layer
Masked multihead attention layer enables decoder to pay attention on previously generated words at inference phase.

However, at training phase, basic Multihead attention layer learns context vector for all words in a sentence, even next words of current word.
To eliminate illegal connection(connection between current word and next words), Masking is applied to Scaled-dot product attention before softmax.

#### 2.4.2. Multihead Attention layer(attention on source sentence)
This layer enables decoder to pay attention on source sentence. 

Let V, K are output of encoder, and Q is an output of Masked multihead attention layer. Q, K, V feeds into Multihead attention layer and computes attention of query Q on source sentence.  

#### 2.4.3. Position-Wise Feed Forward layer
Same as described in 2.3.2.

### 2.5. Training Objective
Training objective is to minimize cross-entropy loss between **predicted translation distribution** and **ground truth translation distribution**, which is same as Seq2Seq training objective.

### 2.6. Advantages of Self Attention
