# Sequence to Sequence learning with Neural Networks
## 1. Background
Conventional deep neural networks can not be used to map sequences to sequences owing to fixed size of input layer. Sequence has unknown length, therefore can not be an input of DNN with fixed input size. 

For example, **Machine translation** is sequential problems as sentence can be expressed as **sequence of tokens**. 
To solve sequential problems, machine translation problem to be specific, authors suggests using LSTM architecture. 

## 2. Methodology
### 2.1. Mathmatical Formulation 

Let input sequence(source sentence) to be denoted as *X = (x<sub>1</sub> ... x<sub>T</sub>)*

Let output sequence(target sentence) to be denoted as *Y = (y<sub>1</sub> ... y<sub>T'</sub>)*

Goal of machine translation is to search sentence Y that maximizes probability *p(*y<sub>1</sub> ... y<sub>T'</sub>* | *x<sub>1</sub> ... x<sub>T</sub>*)*

If source sentence *X* is encoded as fixed size vector *v*, *p(*y<sub>1</sub> ... y<sub>T'</sub>* | *x<sub>1</sub> ... x<sub>T</sub>*)* can be formulated as below according to the law of conditional probability.

<img width="343" alt="20210522_164455" src="https://user-images.githubusercontent.com/70640345/119218775-183c9000-bb1d-11eb-969e-5b553c7bbf05.png">

To sum up, Seq2Seq encoder-decoder model parameterizes probability distribution of **next word**, given source sentence and previously translated word. 

### 2.2. Architecture
Seq2Seq model consists of Encoder and Decoder.

<img width="396" alt="20210522_162429" src="https://user-images.githubusercontent.com/70640345/119218223-44a2dd00-bb1a-11eb-8e1b-51fea4c607c9.png">

1. Encoder is composed of 4 layers LSTM with 1000 dimensional hidden state.  Encoder encodes sequence of source sentence (*x<sub>1</sub> ... x<sub>T</sub>*) to *v*, a fixed size vector. 
2. Decoder is also composed of 4 layers LSTM with 1000 dimensional hidden state. Decoder outputs most probable translation T which maximizes log probability *p(T|S)* using **left to right beam search algorithm**.

#### Beam Search Algorithm

