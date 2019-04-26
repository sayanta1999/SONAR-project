# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 19:59:55 2019

@author: KIIT
"""

import tensorflow as tf
import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split
from  sklearn.utils import shuffle

def read_data():
    df=pd.read_csv('sonar.all-data.csv')
    x=df.iloc[:,0:-1].values
    y1=df.iloc[:,-1]
    y=pd.get_dummies(y1)
    print(y.shape)
    
    
    return x,y


def multilayer_perceptron(x,weights,biases):
    layer_1 = tf.add(tf.matmul(x,weights['w1']),biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1,weights['w2']),biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    
    layer_3 = tf.add(tf.matmul(layer_2,weights['w3']),biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    
    layer_4= tf.add(tf.matmul(layer_3,weights['w4']),biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    
    out_layer=tf.matmul(layer_4,weights['out'])+biases['out']
    
    return out_layer


X,Y=read_data()
X,Y=shuffle(X,Y)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=41)
print(y_train.shape)
n_dim=X.shape[1]
n_classes=2
x = tf.placeholder(tf.float32,shape=[None,n_dim])
#weights = tf.Variable(tf.zeros([n_dim,n_classes]))
#biases = tf.Variable(tf.zeros([n_classes]))

y_=tf.placeholder(tf.float32,shape=[None,n_classes])

epochs=10
learning_rate=0.3
n_hidden1=55
n_hidden2=55
n_hidden3=55
n_hidden4=55

weights={'w1':tf.Variable(tf.truncated_normal([n_dim,n_hidden1])),
         'w2':tf.Variable(tf.truncated_normal([n_hidden1,n_hidden2])),
         'w3':tf.Variable(tf.truncated_normal([n_hidden2,n_hidden3])),
         'w4':tf.Variable(tf.truncated_normal([n_hidden3,n_hidden4])),
         'out':tf.Variable(tf.truncated_normal([n_hidden4,n_classes]))}
biases={'b1':tf.Variable(tf.truncated_normal([n_hidden1])),
        'b2':tf.Variable(tf.truncated_normal([n_hidden2])),
        'b3':tf.Variable(tf.truncated_normal([n_hidden3])),
        'b4':tf.Variable(tf.truncated_normal([n_hidden4])),
        'out':tf.Variable(tf.truncated_normal([n_classes]))}

init=tf.global_variables_initializer()

y=multilayer_perceptron(x,weights,biases)

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess=tf.Session()
sess.run(init)


for epoch in range(epochs):
    sess.run(train_step,feed_dict={x:x_train,y_:y_train})
    cost=sess.run(cost_function,feed_dict={x:x_train,y_:y_train})
    correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    y_pred = sess.run(y,feed_dict={x:x_test})
    mse = tf.reduce_mean(tf.square(y_pred-y_test))
    accuracy = sess.run(accuracy,feed_dict={x:x_train,y_:y_train})
    print("epoch:",epoch,"cost:",cost,"MSE:",sess.run(mse),"train accuracy:",accuracy)
    
#Final Accuracy
correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
final_accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
print("Test Accuracy : ",sess.run(final_accuracy,feed_dict={x:x_test,y_:y_test}))

#Final loss/cost
y_pred = sess.run(y,feed_dict={x:x_test})
mse = tf.reduce_mean(tf.square(y_pred-y_test))
print("Test Loss: ",sess.run(mse))