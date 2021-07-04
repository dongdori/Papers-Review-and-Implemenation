# Effective Approaches to Attention based Neural Machine Translation
## 1. Background
Since sequence to sequence model architecture was introduced, performance of neural machine translation(NMT) improved rapidly.
Attention mechanism also enabled translation model to focus on certain part of source sentence during translation.
Authors suggest two types of attention based NMT model, with effectiveness and simplicity in mind.

## 2. Approach

Neural machine translation system consists of *encoder* and *decoder*.

In seq2seq model, Encoder encodes source sentence into fixed size vector(hidden state of last RNN unit), and Decoder computes conditional probability distribution of **next target word**, given **previously generated target words** and **encoded source sentence**.

To improve performance, decoder have to be able to focus on **specific part of source sentence** during each decoding phase, which is called **Attention mechanism**.

Authors suggest two methodologies for attention mechanism, **Local Attention** and **Global Attention**.

### 2.1. Global Attention
Global attention computes alignment vector for total sequence of source sentence at each translation step. 

1. Let *c<sub>t</sub>* to be **context vector** for t<sub>th</sub> word generation. This context vector feeds into decoder at every translation generation steps instead of last      hidden states of encoder. Context vector are computed as **weighted average of every hidden states at encoder**. Then how the weight can be computed?   

2. Let *a<sub>t</sub>* to be **alignment vector** at t<sub>th</sub> translation generation step. *a<sub>t</sub>* is used as **weights** for computing context vector.  
  *a<sub>t</sub>* is vector length of *S*, which is length of source sentence sequence. 
   Each element of *a<sub>t</sub>* means degree of relativeness between **t<sub>th</sub> translation word** and **each words of input sequence**.  
### 2.2. Local Attention
