#!/usr/bin/python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
import tempfile
from sklearn.model_selection import train_test_split

# importing the dataset
dataset = pd.read_csv("dataset/data_test.csv")
print(dataset.head())
dataset = dataset.drop('id', 1)  # removing unwanted column
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

# spliting the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)


def main():
    # Specify that all features have real-value data
    feature_columns = [tf.compat.v1.estimator.layers.real_valued_column("", dimension=6)]

    # Build 3 layer DNN with 512, 256, 128 units respectively.
    classifier = tf.compat.v1.estimator.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[512, 256, 128],
                                                n_classes=2,
                                                optimizer=tf.train.ProximalAdagradOptimizer(
                                                    learning_rate=0.15,
                                                    l1_regularization_strength=0.001
                                                ))

    # Define the training inputs
    def get_train_inputs():
        x = tf.constant(X_train)
        y = tf.constant(y_train)
        return x, y

    # Fit model.
    classifier.fit(input_fn=get_train_inputs, steps=100)

    # Define the test inputs
    def get_test_inputs():
        x = tf.constant(X_test)
        y = tf.constant(y_test)

        return x, y

    # Evaluate accuracy.
    # print(classifier.evaluate(input_fn=get_test_inputs, steps=1))
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
    graph_location = 'E:/2023 PROJECTS/Website_Phishing_Detection_Android/'
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())
    print("Test Accuracy: {0:f}".format(accuracy_score))

    # Classify two new flower samples.
    # med,med,5more,more,med,high,vgood
    # med,med,4,2,small,high,unacc
    def new_samples():
        return np.array([[-1, -1, -1, 1, -1, -1, 0, -1, 1, -1, 1, 0, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1],[-1, -1, -1, 1, -1, -1, 0, -1, 1, -1, 1, 0, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1]],
                        dtype=np.float32)

    predictions = classifier.predict(input_fn=new_samples)

    print("New Samples, Class Predictions: {}".format(predictions))


if __name__ == "__main__":
    main()
