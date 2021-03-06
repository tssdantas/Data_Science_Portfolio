## Data Science Porfolio.

This repository contains my porfolio of Data Science code created for academic and self-learning purposes.
The main sources used are:
- Swamynathan, M. Mastering Machine Learning with Python in Sex Steps. 2nd Edition. APress. 2019
- Keras Online Documentation
- Tensorflow Online Documentation
- Scikit-Learn Online Documentation
- Pandas & Numpy Documentation


## Contents
- ### Machine Learning
    -  [Supervised Learning](https://github.com/tssdantas/Data_Science_Portfolio/tree/main/Mushroom_classification) An algorithm in Python 3.8.5 is presented to solve the classification problem of mushroom data into the classes of "edible" or "poisonous". The mushroom data contains 22 features/ independent variables and it was obtained from [kaggle.com](https://www.kaggle.com/uciml/mushroom-classification), but it can be found in the same directory as the sample code. The classification methods used are: Random Forest Classifier, Support Vector Machines, K-Nearest Neighbors Voting, Multi-Layer Perceptron Neural Networks and Grandient Boosting Classifier. The Python packages used are: Pandas, Numpy and Scikit-learn. Results are mostly within 99-100% range with the exception of RandomForestClassifier. An approach to reduce False-Postive and False-Negative rates of the RandomForestClassifier is presented in a separete file.
    -  [Deeplearning & Computer Vision](https://github.com/tssdantas/Data_Science_Portfolio/tree/main/SkinCancer) This algorithm uses Keras with Tensorflow backend to train and evaluate a Convolutional Neural Network (CNN) for the image classification problem of skin cancer detection. The target classes are 'benign' and 'malignant' represented in 1800 and 1500 image samples, respectively. The image dataset is not provided here and should be downloaded from [Kaggle.com](https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign). Results: 99% Accuracy with training data and 85% accuracy with validation data.
    -  [Transfer Learning](https://github.com/tssdantas/Data_Science_Portfolio/tree/main/Fashion_MNIST) An approach to implement Transfer Learning is demonstrated with a CNN for the image classification problem. The CNN structure is separated between "features layers" and "classification layers". The first step is to train the CNN with MNIST dataset (Grayscale handwritten digits) achieving 99% accuracy in the validation set. The parameters of the "feature layer" of the CNN obtained in the first step fixed, leaving the parameters in the classification layer free to be ajusted in training and validation with a new dataset: Fashion MNIST ( Grayscale images of cloathing items), resulting in 91% total accuracy. Analysing accuracy in the classification of individual classes ( T-shits (97.8%), Pullover (95%), Coat (94.8%), Shirt (92.2%)) its possible to conclude that all cloathing pieces with lower prediction accuracy are of similar shape and that the hadwritten digits feature layer isnt fully capable of classifying all clothing samples from these classes.

This code is licensed under GNU General Public license v3.0

## Contact information

Any questions should be directed to Tarcisio S. S. Dantas at tssdantas@gmail.com
