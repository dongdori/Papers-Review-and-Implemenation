# Enriching Word Vectors with Subword Information
## 1. Background
Word representations such as Word2vec or GloVe do not consider any morphology of word. It simply assign one vector to one word which is learned by language model.
However, morphologically rich language(Korean, Turkish, Finnish) has limitation with this approach.

Therefore, authors suggest improve vector representations by representing single word as **bag of character n-grams** and represent word vector as a **sum of each vector of character n-grams**.

## 2. Methodology
