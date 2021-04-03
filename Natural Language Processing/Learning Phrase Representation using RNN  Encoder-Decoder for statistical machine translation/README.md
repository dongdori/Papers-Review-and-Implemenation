# RNN Encoder-Decoder for machine translation and Word representation
## 1. Problem Statement and Contents
In the perspective of machine translation, given input sequence, translation model returns output sequence which maximizes conditional probability P(target sentence | source sentence).
The objective of this paper is rescoring phrase pairs through trained RNN Encoder-Decoder and use scores as additional features of statistical machine translation(SMT) model, thereby improving SMT performance.

### Contents
1. RNN Encoder-Decoder Architecture and Explanation of GRU hidden units.
2. Statistical Machine Translation and RNN Encoder-Decoder model as a feature of SMT
---
### Mathmatical Formulation
\[p(x,y)\]
---
## 2. RNN Encoder-Decoder
### RNN Encoder-Decoder Architecture
Let length of source sentence is N, length of output sentece is M.
1. Encoder
- RNN Encoder sequentially reads each token of the input sentence.
- At each timestep, embedding layer transforms token into 100-dimensional embedding vector.
- At each timestep, each hidden units compute hidden state h_<t>, given h_<t-1> and e_<t>, which is embedding vector of word_<t>. (h_<0> and e_<0> are initialized into zero vector)
- Encoder finally computes *1000-dimensional summary vector c* where c = tanh(V*h_<N>).
3. Decoder
- At each timestep,  

