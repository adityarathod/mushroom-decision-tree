# Mushroom Decision Tree

*Created as a learning project within the 
[Clark Summer Research Program](https://honors.utdallas.edu/clark-summer-research-program)
at the University of Texas at Dallas.*

**Disclaimer: This is by no means efficient code. Please don't use in production.**

## About this Model
This model is a multi-branched decision tree algorithm, with each node being able to split on multiple feature values.

The splitting heuristic is information gain, as opposed to Gini impurity.

Information gain is calculated as <img src="http://www.sciweavers.org/upload/Tex2Img_1560811146/render.png" />, where
H(Y) is the entropy of the variable Y and H(Y|X) is the *conditional* entropy of the variable Y given X.

## Performance
Surprisingly, this model performs with 100% accuracy on the test set. Possible reasons for this may include:

- extreme overfitting
- having input data that properly represents real-world input