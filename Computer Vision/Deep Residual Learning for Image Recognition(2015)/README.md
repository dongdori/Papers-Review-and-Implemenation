# Deep Residual Learning for Image Recognition
## 1 . Problem statement
Deep convolutional neural nerwork brought state of art performance on image classification task, nevertheless suffers from vanishing gradient problem.
Owing to vanishing gradient, accuaracy degradation occurs, which is not caused by overfitting.
Author suggests short cut connection between layers, which enables Residual Learning. With residual learning, deep convolutional neural network do not suffers from vanishing gradient problem and able to gain accuracy. 

## 2. Formulation of Residual Learning
- Let H(x) as un underlying mapping which have to be fitted by a few layers. 
- Let F(x) = H(x) - x to be a residual from original mapping. 

What if H(x) = x, which means H(x) have to be identity mapping?
A few non-linear layers have difficulties on approximating identity mapping and degradation problem occurs.

However when H(x) = x, F(x) = 0 and F(x) = 0 can be easily fitted by a few non-linear layers.

Therefore, authors suggest adopting residual learning to every few stacked layers.

Each building blocks are defined as below
1. x : input vectors(feature maps) of layers
2. y : output vectors(feature maps) of layers
3. y is formulated as y = F(x,{W<sub>i</sub>}) + x, where x is added by element-wise addition.


