# Enriching Word Vectors with Subword Information
## 1. Background
Word representations such as ```Word2vec``` or ```GloVe``` do not consider any **morphology** of word. It simply assign one vector to one word which is learned by language model.
However, morphologically rich language(Korean, Turkish, Finnish) has limitation with this approach.

Therefore, authors suggest improve vector representations by representing single word as **bag of character n-grams** and represent word vector as a **sum of each vector of character n-grams**.

## 2. Methodology

### 2.1. Skip gram with Negative Sampling
This paper utilizeS Skip-gram language model with negative sampling. 

When training word representation without information of subword, Optimization objective is as below.

<img width="218" alt="20210520_211426" src="https://user-images.githubusercontent.com/70640345/118977095-c119ac80-b9b0-11eb-90a8-7e3b936740c2.png">

where **function l(x) = log(1 + exp(-x))** and function **s(w<sub>t</sub>, w<sub>c</sub>)** is scoring function **(scalar product between word vector w<sub>t</sub> and w<sub>c</sub>)** 

### 2.2. Learning vector representation of subword
1. Character n-gram
* To learn vector representation of subword, we first have to represent single word as **bag of character n-grams**.
* For example, for word **professor**, we can decompose it as tri-gram like {**<pr, pro, rof, ofe, fes, ess, sso, sor, or>**}
* In practice, n is set to be 3~6.

2. Modified optimization objective for subword information
* Given a word *w*, let us set of n-grams of word *w* denoted as g<sub>w</sub> = {1,...G}.
* Futhermore, let us denote vector representation of n-grams 1...G as z<sub>1</sub>... z<sub>G</sub>.

Then scoring function for word *w* and word *c* is as below.

<img width="106" alt="20210520_213824" src="https://user-images.githubusercontent.com/70640345/118979752-c5939480-b9b3-11eb-9f1e-fe304704411d.png">

When vector representations of subwords are sufficiently learned, We can define vector representation of word *w* as simply

<b><i>v<sub>w</sub></i> = SUM(z<sub>1</sub>, ... , z<sub>G</sub>)</b>

## 3. Result
1. Subword modeling resulted in better word vector representation, especially for morphologically rich language.

2. Also subword vector representation enable us to define word representation vector for Out-Of-Vocabulary words. For example, let *transform* is OOV word, but if we have representation vector of subwords *trans* and *form*, We can just add them up to define word vecetor of *transform*!

Therefore, we can express rare words which are OOV, by summing subword vector.

3. Most interesting result was that **Morpheme(형태소)** of each word can be identified through analysis on subword representation vector and word representation vector.

Let z<sub>1</sub>...z<sub>G</sub> denotes subword vectors of word w, *v<sub>w</sub>* denotes vector representation of word w

Let v<sub>w-g</sub> denotes v<sub>w</sub> - z<sub>g</sub>

If cosine distance between v<sub>w-g</sub> and v<sub>w</sub> is large, It implies subword g plays a crucial role in word w. Table below illustrates most important n-grams(subwords) in given words.

<img width="235" alt="20210521_172513" src="https://user-images.githubusercontent.com/70640345/119106752-959ecc80-ba59-11eb-9ba4-d9136fb779ec.png">




 
