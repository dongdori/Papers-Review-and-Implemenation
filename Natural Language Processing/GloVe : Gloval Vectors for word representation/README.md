# GloVe : Gloval Vectors for word representation

## 1. Problem Statement
There are two main approaches to learn word vector representation.

1. Supervised method - Local context window methods
  - Through training skip-gram model(predicting word given context words) or CBOW model(predicting contexts word given some word), *Embedding layer* learns appropriate representation of words
  - Advantage : Good performance on *word analogy tasks*, which means linear relationships between words are effectively learned.
  - Disadavantage : Gloval co-occurence statistics in corpus are not taken into account 
2. Unsupervised method - Word co-occurence Matrix Factorization based methods
  - Exploiting word co-occurence matrix X generated from entire corpus. *X<sub>i,j</sub>* denotes occurence of *word<sub>j</sub>* in context of *word<sub>i</sub>*
  - Advantage : *Entire word count statistics* are taken into account
  - Disadvantage : Poor performance on word analogy task relative to Local context window methods

To offset disadvantages of each approach, Authors suggest supervised method based on word co-occurence matrix, which is form of log-bilinear model

## 2. Approach
### 1. Build word co-occurence matrix.
  Based on certain size of widow, we have to build word co-occurence matrix X.
  X<sub>i,j</sub> denotes how many times word<sub>j</sub> appeared in context of word<sub>i</sub>. Therefore, X<sub>i,j</sub> = X<sub>j,i</sub>, which implies X is symmetrical matrix.
  
### 2. Setting target values from word co-occurence matrix
  GloVe is different from matrix factorization based methods, because GloVe representation vectors learned by model with supervision. Therefore, appropriate target
  value have to be set. Let's look at formulas in co-occurence matrix and what they imply.
  - *X<sub>i,j</sub>*: occurence of word j in context of word i
  - *X<sub>i</sub>* : occurence of any words in context of word i
  - *X<sub>i,j</sub>* = *X<sub>i,j</sub> / X<sub>i</sub>* : probability that word j appears in context of word i (Co-occurence probability)
  - *P<sub>i,k</sub>/ P<sub>j,k</sub>* : degree of likelihood that word i appears in context of word k rather than word j appears in context of word k (Ratio of co-occurence probability)  

Authors set **ratio of Co-occurence probability** as target of GloVe model.

### 3. Cost Function
Let *w<sub>i</sub>* denotes embedding vector of word<sub>i</sub>.
As I said earlier, target of this model is to predict *P<sub>i,k</sub>/ P<sub>j,k</sub>* ,given word vectors. More accurately, The goal is **to get learned word vector representations** through training model to accurately predict *P<sub>i,k</sub>/ P<sub>j,k</sub>*.

Cost function is defined as below. 

![image01](https://user-images.githubusercontent.com/70640345/115567005-c0c8bb80-a2f5-11eb-9572-9cc0b8449017.png)



 
