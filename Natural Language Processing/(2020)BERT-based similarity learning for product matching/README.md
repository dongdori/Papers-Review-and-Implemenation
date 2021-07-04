# Paper Review - BERT based similarity learning for product matching

For product matching task, we can make use of BERT encoder for similarity matching. According to many research, It has been discovered that 
Fine tuning BERT with task-specific data can improve model performance.   You might read https://www.aclweb.org/anthology/2020.ecomnlp-1.7.pdf.

## 1. Problem Statement
Many retailors upload their products to sell via E-Commerce site. In the perspective of E-commerce company, Matching same product is crucial for improving User Experience.
We can make use of Image, product description text data to develop deep learning algorithm for Product Matching.   In this repository, Through computing sentence embeddings of each product and caculate cosine distance of each of them, I will match similar or same product.

## 2. Model Architecture
You can see BERT encoder architecture below. 
<img width="857" alt="20210322_073446" src="https://user-images.githubusercontent.com/70640345/111923177-4b44b200-8ae1-11eb-8e37-715d3667252e.png">
1. Sentence is tokenized into integer and feeds into BERT. 
2. BERT outputs Tensor shape of *(batch_size, #tokens in the sentence, dimension of word embeddings)* which means, word embeddings of each words in the sentence.
3. Avarage pooling layer and Max pooling layer pools each word embeddings therefore outputs tensor shape of *(batch_size, dimension of word embeddings)*
4. Max pooled output and Avarage pooled output are concatenated which is tensor shape of *(batch_size, 2 * dimension of word embeddings)*
5. concatenated vector passes through Linear Transformation layer thereby outputs *Sentence Embeddings*

If we employ pretrained BERT output and pooling them without any fine tuning, model performance is even worse than just averaging GloVe embedding vectors, which do not reflect sentence context. Therefore, for better performance, It is recommended to Fine Tune *BERT* and *linear transformation layer* on top of pooling layer with Task Specific Data.
There are a few ways to fine tune BERT for similarity Matching. I tried to optimize *Triplet Loss* through Siamese Network Architecture therefore updating parameters of BERT and Linear Transformation Layer. 

## 3. Triplet Loss
Please refer to link below if you want to learn Triplet Loss and Triplet Mining Strategies.   https://omoindrot.github.io/triplet-loss#compute-the-distance-matrix
I chose *Batch-SemiHard Strategy* which is one of the most popular triplet mining strategies.  

## 4. Training Siamese Network by optimizing Triplet Loss
<img width="934" alt="20210321_231326" src="https://user-images.githubusercontent.com/70640345/111908226-c6837500-8a9b-11eb-951a-ce9ddd34bf37.png">
Figure above describes architecture of Siamese Network.   
*Three BERT and Three Linear Transformation Layer are tied, which means they always share same parameter*
