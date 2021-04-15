# RNN Encoder-Decoder for machine translation and Word representation
## 1. Problem Statement and Contents
In the perspective of machine translation, given input sequence, translation model returns output sequence which maximizes conditional probability

<img width="158" alt="20210412_163214" src="https://user-images.githubusercontent.com/70640345/114357493-c400d080-9bac-11eb-9c82-4e606d2cd842.png">

where y_n indicates target sentence and x_n indicates source sentence.

The objective of this paper is rescoring phrase pairs through trained RNN Encoder-Decoder and use scores as additional features of statistical machine translation(SMT) model, thereby improving SMT performance.

---
Since RNN-Encoder Decoder is trained, It can be used not only to rescore translation given source sentence and target sentence, but also to generate target sentence given source sentence. However, Author left this for future work as it takes expensive sampling procedure.

Auther suggests to use RNN-Encoder Decoder for scoring pharase pair thereby use scores as feature in SMT.

However, In this review, I will focus on RNN-Encoder Decoder architecture and mathmatical principals rather than its application to SMT.

---
## 2. RNN Encoder-Decoder Architecture
Let length of source sentence is N, length of output sentece is M.
### 1. Encoder
- RNN Encoder sequentially reads each token of the input sentence.
- At each timestep, embedding layer transforms token into 100-dimensional embedding vector.
- At each timestep, each hidden units compute hidden state, given previous hidden state and current embedding vector.

<img width="250" alt="encoder hidden state" src="https://user-images.githubusercontent.com/70640345/114360994-a03f8980-9bb0-11eb-84cb-000447f74f33.png">

Weight matrix W and U are dimension of 1000x100

- Encoder finally computes *1000-dimensional summary vector c* through computation below.

<img width="118" alt="summary vector" src="https://user-images.githubusercontent.com/70640345/114361237-dc72ea00-9bb0-11eb-95c7-208b64adfdb5.png">

### 2. Decoder
 

## 3. Hidden Units
Author suggests unique hidden unit that can drop unnecessary memory and even less complex than LSTM cell. 
