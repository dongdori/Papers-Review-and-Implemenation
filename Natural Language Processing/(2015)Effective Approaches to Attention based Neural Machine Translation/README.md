# Effective Approaches to Attention based Neural Machine Translation
## 1. Background
Since sequence to sequence model architecture was introduced, performance of neural machine translation(NMT) improved rapidly.
Attention mechanism also enabled translation model to focus on certain part of source sentence during translation.
Authors suggest two types of attention based NMT model, with effectiveness and simplicity in mind.

## 2. Approach

Neural machine translation system consists of ```encoder``` and ```decoder```.

In ```seq2seq``` model, Encoder encodes source sentence into fixed size vector(hidden state of last RNN unit), and Decoder computes conditional probability distribution of **next target word**, given **previously generated target words** and **encoded source sentence**.

To improve performance, decoder have to be able to focus on **specific part of source sentence** during each decoding phase, which is called **Attention mechanism**.

Authors suggest two methodologies for attention mechanism, **Local Attention** and **Global Attention**.

### 2.1. Global Attention
Global attention computes alignment vector *a<sub>t</sub>* for total sequence of source sentence at each decoding step t. 

1. Let *a<sub>t</sub>* denotes **alignment vector** at t<sub>th</sub> translation generation step. It means **degree of attention** that t<sub>th</sub> translation pays on each part of source sentence.

2. *a<sub>t</sub>* is vector length of *S*, which is length of source sentence sequence. *a<sub>t</sub>* is used as **weights** for computing context vector *c<sub>t</sub>*.
 *c<sub>t</sub>* is computed as ```tf.matmul([hidden_states, a_t])``` where ```hidden_states``` is defined as ```np.hstack([hidden_state_1, ..., hiddens_state_s])```.

### 2.2. Local Attention
