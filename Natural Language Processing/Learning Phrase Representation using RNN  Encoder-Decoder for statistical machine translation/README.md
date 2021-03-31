# RNN Encoder-Decoder for machine translation and Word representation
## 1. Problem Statement
In the perspective of machine translation, given input sequence, model have to return output sequence of maximum conditional probability 'P(output sentence | input sentence)'.
Main Idea of this paper is that, RNN-encoder network transforms input sequence of varying-length to fixed-length vector. and then, RNN-decoder network generates output sequence based on encoded fixed-length vector.  
