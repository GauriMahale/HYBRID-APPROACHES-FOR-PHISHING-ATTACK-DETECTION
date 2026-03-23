# importing libraries
import joblib # parellel computing in python
import tensorflow as tf #building and training machine learning models.
import matplotlib.pyplot as plt # plot build
import pandas as pd # anlysis of dataset
from keras.layers import Dense
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier, VotingClassifier # multiple decision trees and combines their predictions.
from sklearn.linear_model import LogisticRegression # best fitting model that can predict the probability of an instance belonging to a particular class.
from sklearn.metrics import confusion_matrix #evaluation metric for classification models.
from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_curve, auc, roc_auc_score
# from matplotlib import pyplot
from sklearn.naive_bayes import GaussianNB #probabilistic classifier based on Bayes' theorem
from sklearn.svm import SVC  # hyper plane
from xgboost import XGBClassifier
import elm

cm1 = []
cm2 = []
cm3 = []
cmt = []

# importing the dataset
dataset = pd.read_csv("dataset/phishcoop.csv")
print(dataset.head())
dataset = dataset.drop('id', 1)  # removing unwanted column
x = dataset.iloc[:, :-1].values # all rows and all columns except the last column
y = dataset.iloc[:, -1:].values# all rows and only the last column

# spliting the dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

# # GaussianNB
NB_classifier = GaussianNB()
NB_classifier.fit(x_train, y_train)
# predicting the tests set result
y_pred = NB_classifier.predict(x_test)
# confusion matrix
cm1 = confusion_matrix(y_test, y_pred)

# # Support Vector Machine
# applying grid search to find best performing parameters

parameters = [{'C': [0.1, 0.5, 1, 5, 10], 'gamma': [0.1, 0.2, 0.3, 0.5, 0.6, 0.7]}]#leading to a smaller margin (lower bias, higher variance).points close to the decision boundary have more impact
svm_classifier = SVC(C=5, kernel='rbf', gamma=0.2, random_state=0, class_weight='balanced', probability=True)#radial basis function,shape of the decision boundary
svm_classifier.fit(x_train, y_train)
# predicting the tests set result
y_pred = svm_classifier.predict(x_test)
# confusion matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_pred)
svm_classifier.score(x_train, y_train)
svm_classifier.score(x_test, y_test)

#classifier is able to generalize to unseen data and whether it is overfitting or underfitting the training data.

# Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC)
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot

probs = svm_classifier.predict_proba(x_test)# predicted class probabilities
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
pyplot.show()


def elm_execution():
    # built model and train
    model = elm.elm(hidden_units=64, activation_function='relu', random_type='normal', x=x_train, y=y_train, C=0.1,
                    elm_type='clf')
    beta, train_accuracy = model.fit('solution1')
    print("ELM classifier train accuracy:", train_accuracy)
    prediction = model.predict(x_test)
    print('ELM classifier test accuracy:', model.score(x_test, y_test))
    return model.score(x_test, y_test)


# Voting classifier

from sklearn.ensemble import GradientBoostingClassifier

# Create a Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, random_state=0)

# Create the three classifiers
clf1 = LogisticRegression(max_iter=1000)

# rf
rf_classifier = RandomForestClassifier(n_estimators=200, criterion="entropy", min_samples_split=20, max_features='sqrt',
                                       random_state=0,
                                       class_weight="balanced", warm_start=True)

svm_classifier = SVC(C=5, kernel='rbf', gamma=0.2, random_state=0, class_weight='balanced', probability=True)
from xgboost import XGBClassifier

# Create an XGBoost classifier
xgb_classifier = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0)

from sklearn.neural_network import MLPClassifier

# Create a multi-layer perceptron classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, max_iter=1000,
                               random_state=0)


from sklearn.neighbors import KNeighborsClassifier

# Create a K-nearest neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)


# Create a voting classifier with the three models
voting_classifier = VotingClassifier(
    estimators=[('lr', clf1), ('knn', knn_classifier), ('mlp', mlp_classifier), ('rf', rf_classifier),
                ('xgb', xgb_classifier), ('svm', svm_classifier), ('gbc', gb_classifier)], voting='hard')

voting_classifier.fit(x_train, y_train)
# Fit the voting classifier on the new training data

# Predict the test set labels using the logistic regression model
y_pred = voting_classifier.predict(x_test)

# Calculate the confusion matrix
cm_voting = confusion_matrix(y_test, y_pred)

################

rf_classifier = RandomForestClassifier(n_estimators=200, criterion="entropy", min_samples_split=20, max_features='sqrt',
                                       random_state=0,
                                       class_weight="balanced", warm_start=True)
rf_classifier.fit(x_train, y_train)
# predicting the tests set result
y_pred = rf_classifier.predict(x_test)
# confusion matrix
cmt = confusion_matrix(y_test, y_pred)

# Fit the voting classifier on the training data

# Predict the test set labels using the voting classifie

# Print the accuracy and specificity of the voting classifier
print("Voting Classifier")
accuracy = (cm_voting[0][0] + cm_voting[1][1]) / sum(sum(cm_voting))
specificity = cm_voting[0][0] / (cm_voting[0][0] + cm_voting[0][1])
print("Voting classifier accuracy: {:.2f}%".format(accuracy * 100))
print("Voting classifier specificity: {:.2f}%".format(specificity * 100))

print("NB")
print("Accuracy", (float(cm1[0][0]) / (float(cm1[0][0]) + float(cm1[1][0]))) * 100)
print("Specificity", float(cm1[1][1]) / (float(cm1[0][1]) + float(cm1[1][1])))

print("SVC")
print("Accuracy", (float(cm3[0][0]) / (float(cm3[0][0]) + float(cm3[1][0]))) * 100)
print("Specificity", float(cm3[1][1]) / (float(cm3[0][1]) + float(cm3[1][1])))

print("RF")
print("Accuracy", (float(cmt[0][0]) / (float(cmt[0][0]) + float(cmt[1][0]))) * 100)
print("Specificity", float(cmt[1][1]) / (float(cmt[0][1]) + float(cmt[1][1])))

elm_accuraccy = elm_execution()

# x-coordinates of left sides of bars
left = [1, 2, 3, 4, 5]

# heights of bars
height = [(float(cm1[0][0]) / (float(cm1[0][0]) + float(cm1[1][0]))) * 100,
          (float(cm3[0][0]) / (float(cm3[0][0]) + float(cm3[1][0]))) * 100,
          (float(cm_voting[0][0]) / (float(cm_voting[0][0]) + float(cm_voting[1][0]))) * 100,
          (float(cmt[0][0]) / (float(cmt[0][0]) + float(cmt[1][0]))) * 100,
          elm_accuraccy * 100]
print(len(height))
# labels for bars
tick_label = ['NB', 'SVM', 'voting', 'RF', 'ELM']

# plotting a bar chart
plt.bar(left, height, tick_label=tick_label,
        width=0.8, color=['red', 'green', 'yellow', 'blue', 'black'])

# naming the x-axis
plt.xlabel('x - axis')
# naming the y-axis
plt.ylabel('y - axis')
# plot title
plt.title('Accuracy Visualization!')

# function to show the plot
plt.show()

# model specification


joblib.dump(voting_classifier, 'classifier/model.pkl')
