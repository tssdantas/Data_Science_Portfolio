import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 


#from sklearn.metrics import confusion_matrix
df = pd.read_csv('mushrooms.csv')
y = df['class']
X = df.iloc[:,1:]

# replace() remaps variables in the dataframe from a one letter discription to a string, 
# defined in the dictionary 'mapping'.... for example: from 'b' to 'bell' 
mapping = { 'cap-shape': {'b': 'bell', 'c': 'conical', 'f': 'flat', 'k': 'knobbed', 'x': 'convex', 's' : 'sunken'},
            'cap-surface': {'y': 'scaly', 's': 'smooth', 'f': 'fibrous', 'g': 'grooves', },
            'cap-color': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'r': 'green', 'p':'pink',
                          'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'},
            'bruises': {'t': 'bruises', 'f': 'no'},
            'odor': {'a': 'almond', 'l': 'anise', 'c': 'creosote', 'y': 'fishy', 'f': 'foul', 'm': 'musty', 
                     'n': 'none', 'p': 'pungent', 's': 'spicy'},
            'gill-attachment': {'a': 'attached', 'f': 'free'},
            'gill-spacing': {'c': 'close', 'w': 'crowded'},
            'gill-size': {'b': 'broad', 'n': 'narrow'},
            'gill-color': {'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'g': 'gray', 'r': 'green',
                           'o': 'orange', 'p': 'pink', 'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'},
            'stalk-shape': {'e': 'enlarging', 't': 'tapering'},
            'stalk-root': {'b': 'bulbous', 'c': 'club', 'u': 'cup','e': 'equal', 'z': 'rhizomorphs', 
                           'r': 'rooted', '?': 'missing'},
            'stalk-surface-above-ring': {'f': 'fibrous', 'y': 'scaly', 'k': 'silky', 's': 'smooth'},
            'stalk-surface-below-ring': {'f': 'fibrous', 'y': 'scaly', 'k': 'silky', 's': 'smooth'},
            'stalk-color-above-ring': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange', 
                                       'p': 'pink', 'e': 'red', 'w': 'white', 'y': 'yellow'},
            'stalk-color-below-ring': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange', 
                                       'p': 'pink', 'e': 'red', 'w': 'white', 'y': 'yellow'},
            'veil-type': {'p': 'partial', 'u': 'universal'},
            'veil-color': {'n': 'brown', 'o': 'orange', 'w': 'white', 'y': 'yellow'},
            'ring-number': {'n': 'none', 'o': 'one', 't': 'two'},
            'ring-type': {'c': 'cobwebby', 'e': 'evanescent', 'f': 'flaring', 'l': 'large', 'n': 'none', 
                          'p': 'pendant', 's': 'sheathing', 'z': 'zone'},
            'spore-print-color': {'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'r': 'green', 
                                  'o': 'orange', 'u': 'purple', 'w': 'white', 'y': 'yellow'},
            'population': {'a': 'abundant', 'c': 'clustered', 'n': 'numerous', 's': 'scattered', 
                           'v': 'several', 'y': 'solitary'},
            'habitat': {'g': 'grasses', 'l': 'leaves', 'm': 'meadows', 'd': 'woods', 'p': 'paths', 
                        'u': 'urban', 'w': 'waste'},
            'class': {'e': 'edible', 'p':'poisonous'}
          }
df.replace(mapping, inplace = True)


#Transform data from label (string/object) to numeric (integer), see "mapping"
encoder = OneHotEncoder(handle_unknown='ignore')
X = encoder.fit_transform(X)

#Transform data from label (string/object) to numeric (integer), edible = 1, poisonous = 0
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#spliting data into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#Supervised methods
#----- Boosting 
# Freud and Schapire in 1995 introduced the concept of boosting with the well-known
# AdaBoost algorithm (adaptive boosting). The core concept of boosting is that rather
# than an independent individual hypothesis, combining hypotheses in a sequential
# order increases the accuracy. Essentially, boosting algorithms convert the weak
# learners into strong learners. Boosting algorithms are well designed to address bias
# problems
# Due to the stagewise additivity, the loss function can be represented in a form suitable
# for optimization. This gave birth to a class of generalized boosting algorithms known as
# generalized boosting machine (GBM). Gradient boosting is an example implementation
# of GBM and it can work with different loss functions such as regression, classification,
# risk modeling, etc. As the name suggests, it is a boosting algorithm that identifies
# shortcomings of a weak learner by gradients (AdaBoost uses high-weight data points),
# hence the name gradient boosting.
# Model complexity and overfitting can be controlled by using correct values for first inputs
clf_a = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf_a.fit(X_train, y_train)
y_pred_a = clf_a.predict(X_test)
print('Accuracy_score GradientBoostingClassifier: ', accuracy_score(y_test, y_pred_a))

# Regular Supervised methods
clf_b = RandomForestClassifier(max_depth=2, random_state=0)
clf_b.fit(X_train, y_train)
y_pred_b = clf_b.predict(X_test)
print('Accuracy_score RandomForestClassifier : ', accuracy_score(y_test, y_pred_b))

clf_c = SVC(kernel='linear', C=1.0, random_state=0)
clf_c.fit(X_train, y_train)
y_pred_c = clf_c.predict(X_test)
print('Accuracy_score SVM : ', accuracy_score(y_test, y_pred_c))

clf_d = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski' )
clf_d.fit(X_train,y_train)
y_pred_d = clf_d.predict(X_test)
print('Accuracy_score KNeighborsClassifier : ', accuracy_score(y_test, y_pred_d))


#ANN (Supervised)
# Initialize ANN classifier
mlp = MLPClassifier(hidden_layer_sizes=(20), activation='logistic',max_iter = 100)
# Train the classifier with the training data
mlp.fit(X_train,y_train)

MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
hidden_layer_sizes=(30, 30, 30), learning_rate='constant',
learning_rate_init=0.001, max_iter=100, momentum=0.9,
nesterovs_momentum=True, power_t=0.5, random_state=None,
shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
verbose=False, warm_start=False)
# print("Training set score: %f" % mlp.score(X_train, y_train))
# print("Test set score: %f" % mlp.score(X_test, y_test))

#predict results from the test data
y_pred_ann = mlp.predict(X_test)
print('Accuracy_score ANN : ', accuracy_score(y_test, y_pred_ann))

#To use Numpy´s concatenate() it is required to transform 1D arrays --> 2D arrays   ( (n) --> (n,1) )  
print('\nReshaping 1D arrays into 2D for concatenate() function ... \n')
print('\nDimension of 1D arrays: ', y_test.shape, y_pred_ann.shape, y_pred_d.shape, y_pred_c.shape, y_pred_b.shape, y_pred_a.shape )
y_test = np.reshape(y_test, [-1,1])
y_pred_ann = np.reshape(y_pred_ann, [-1,1])
y_pred_d = np.reshape(y_pred_d, [-1,1])
y_pred_c = np.reshape(y_pred_c, [-1,1])
y_pred_b = np.reshape(y_pred_b, [-1,1])
y_pred_a = np.reshape(y_pred_a, [-1,1])
print('\nDimension of 2D arrays: ', y_test.shape, y_pred_ann.shape, y_pred_d.shape, y_pred_c.shape, y_pred_b.shape, y_pred_a.shape )

my_array = np.concatenate( (y_test, y_pred_ann, y_pred_d, y_pred_c, y_pred_b, y_pred_a), axis=1  )
resultdf = pd.DataFrame(my_array, columns = ['True Result','Pred. Ann', 'Pred. KNeighbors', 'Pred. SVM','Pred. Rand.Forest' , 'Pred. Grad.Boost'] )
resultdf.to_csv('results.csv', index=False)
