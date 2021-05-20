# Enriching Word Vectors with Subword Information
## 1. Background
Word representations such as Word2vec or GloVe do not consider any morphology of word. It simply assign one vector to one word which is learned by language model.
However, morphologically rich language(Korean, Turkish, Finnish) has limitation with this approach.

Therefore, authors suggest improve vector representations by representing single word as **bag of character n-grams** and represent word vector as a **sum of each vector of character n-grams**.

## 2. Methodology

### 2.1. CBOW with Negative Sampling
This paper utilizeS CBOW language model with negative sampling. 

When training word representation without information of subword, Optimization objective is as below.
<img width="218" alt="20210520_211426" src="https://user-images.githubusercontent.com/70640345/118977095-c119ac80-b9b0-11eb-90a8-7e3b936740c2.png">

where **function l(x) = log(1 + exp(-x))** and function **s(w<sub>t</sub>, w<sub>c</sub>)** is scoring function **(scalar product between word vector w<sub>t</sub> and w<sub>c</sub>)** 

### 2.2. Learning vector representation of subword

