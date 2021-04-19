# GloVe : Gloval Vectors for word representation

## 1. Problem Statement
There are two main approach for generating word representation.

1. Supervised method - Local context window methods
  - Through training skip-gram model(predicting word given context words) or CBOW model(predicting contexts word given some word), Embedding layer learns appropriate representation of words
  - Advantage : Good performance on word analogy tasks, which means linear relationships between words are effectively learned.
  - Disadavantage : Gloval co-occurence statistic in corpus is not taken into account 
2. Unsupervised method - Word co-occurence Matrix Factorization
  - Generating word co-occurence matrix X from entire corpus. X_{i,j} denotes occurence of word_{j} in context of word_{i}
  - Advantage : Entire word count statistics are taken into account
  - Disadvantage : Poor performance on word analogy task relative to Local context window methods

To offset disadvantages of each approach, Authors suggest supervised method based on word co-occurence matrix, which is form of log-bilinear model

## 2. Approach

 
