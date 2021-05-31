# Effective Approaches to Attention based Neural Machine Translation
## 1. Background
Since sequence to sequence model architecture was introduced, performance of neural machine translation(NMT) improved rapidly.
Attention mechanism also enabled translation model to focus on certain part of source sentence during translation.
Authors suggest two types of attention based NMT model, with effectiveness and simplicity in mind.

## 2. Approach

Neural machine translation system consists of *encoder* and *decoder*.

In seq2seq model, Encoder encodes source sentence into fixed size vector(hidden state of last RNN unit), and Decoder computes conditional probability distribution of **next target word**, given **previously generated target words** and **encoded source sentence**.

To improve performance, decoder have to be able to focus on **specific part of source sentence** during each decoding phase, which is called **Attention mechanism**.

Authors suggest two methodologies for attention mechanism.

### 2.1. Global Attention


### 2.2. Local Attention
