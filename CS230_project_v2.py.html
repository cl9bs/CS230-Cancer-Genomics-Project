
# coding: utf-8

# In[76]:


import math
import numpy as np
from sklearn import linear_model, metrics
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.python.framework import ops
#from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

get_ipython().magic('matplotlib inline')
np.random.seed(1)


# In[77]:


##load the data to pandas dataframe
data = pd.read_table('HiSeqV2')
print (data.info())

##visualize the first few rows of the data
#print (data.iloc[1:8])
##clean the data
data_clean = data.select_dtypes(include=['float64'])
data_clean = data_clean.drop(['TCGA-E2-A15A-06', 'TCGA-BH-A18V-06',  'TCGA-E2-A15K-06',  'TCGA-AC-A6IX-06', 'TCGA-E2-A15E-06', 'TCGA-BH-A1ES-06', 'TCGA-BH-A1FE-06'], axis=1)
column_names = list(data_clean.columns.values)
label = [s[-2:] for s in column_names]
label = [1 if s == '01' else 0 for s in label]



##split the data randomly into training and test


train_index = random.sample(range(0, len(label)), 800)
train_x = [data_clean.iloc[:,j] for j in train_index]
train_y = [label[j] for j in train_index]
test_x = [data_clean.iloc[:,j] for j in range(0, len(label)) if j not in train_index]
test_y = [label[j] for j in range(0, len(label)) if j not in train_index]




# ## Model 1 : Logistic Regression

# In[79]:


##logistic regression
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(train_x, train_y)
z = logreg.predict(test_x)

#precision, recall, F-value
precision = metrics.precision_score(test_y, z, labels=None, pos_label=1, average='binary', sample_weight=None)
print(precision)

recall = metrics.recall_score(test_y, z, labels=None, pos_label=1, average='binary', sample_weight=None)
print(recall)

F1_score = metrics.f1_score(test_y, z, labels=None, pos_label=1, average='binary', sample_weight=None)
print(F1_score)




# ## Model 2: Neural network with TensorFlow
# 

# In[80]:


## turn the data into the right shape
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)


train_y = train_y.reshape((train_y.shape[0], 1))
test_y = test_y.reshape((test_y.shape[0], 1))



train_x = np.transpose(train_x)
train_y = np.transpose(train_y)
test_x = np.transpose(test_x)
test_y = np.transpose(test_y)


# In[81]:


##create placeholders
def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an feature vector (20530 genes)
    n_y -- scalar, number of classes
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    X = tf.placeholder(tf.float32, shape=[n_x, None])
    Y = tf.placeholder(tf.float32, shape=[n_y, None])
    
    return X, Y


# In[82]:


X, Y = create_placeholders(20530, 1)
print ("X = " + str(X))
print ("Y = " + str(Y))


# In[83]:


##initialize_parameters
def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 20530]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [1, 12]
                        b3 : [1, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [25, 20530], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1, 12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [1,1], initializer = tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


# In[84]:


tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


# In[85]:


##forward propagation
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                                            # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                            # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                                              # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                                               # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###
    
    return Z3


# In[86]:


tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(20530, 1)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))


# In[87]:


##compute cost

def compute_cost(Z3, Y):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    targets -- vector of labels y (1 or 0) 
    
    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels" 
    in the TensorFlow documentation. So logits will feed into z, and labels into y. 
    
    Returns:
    cost -- runs the session of the cost (formula (2))
    """
     
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = Z3
    targets = Y
    
    # Use the loss function (approx. 1 line)
    cost = tf.nn.weighted_cross_entropy_with_logits(targets = targets, logits = logits, pos_weight = 0.3)
    
    return cost


# In[88]:


tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(20530, 1)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Y, Z3)
    print("cost = " + str(cost))


# In[89]:


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_iterations = 15000, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X_train -- training set, of shape (input size = 20530, number of training examples = 800)
    Y_train -- test set, of shape (output size = 1, number of training examples = 800)
    X_test -- training set, of shape (input size = 20530, number of training examples = 411)
    Y_test -- test set, of shape (output size = 1, number of test examples = 411)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    

    # Initialize parameters
    parameters = initialize_parameters()
    
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
    
    
    # L1 regularization
    l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
    weights = tf.trainable_variables() # all vars of your graph
    regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
    regularized_loss = cost + regularization_penalty # this loss needs to be minimized

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(regularized_loss)
    
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Run the session to execute the "optimizer" and the "cost"
        _ , batch_cost = sess.run((optimizer, cost), feed_dict = {X: X_train, Y: Y_train})
                 
#         # plot the cost
#         plt.plot(np.squeeze(batch_cost))
#         plt.ylabel('cost')
#         plt.xlabel('iterations (per tens)')
#         plt.title("Learning rate =" + str(learning_rate))
#         plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        sigmoid_Z3 = tf.sigmoid(Z3) 
        predictions = tf.squeeze(tf.cast(tf.map_fn(lambda x: tf.round(x), sigmoid_Z3), tf.int32))
        
        
        # Precision, recall
        Y_test_placeholder = tf.placeholder(tf.int32, shape=[411]) 
        predictions_placeholder = tf.placeholder(tf.int32, shape=[411])
        precision, precision_update_op = tf.metrics.precision(labels = Y_test_placeholder, predictions = predictions_placeholder)
        
        recall, recall_update_op = tf.metrics.recall(labels = Y_test_placeholder, predictions = predictions_placeholder)
        sess.run(tf.local_variables_initializer())
        final_predictions = sess.run(predictions, feed_dict={X: X_test})
        print(final_predictions)
        print(np.squeeze(Y_test))
        precision_value, recall_value = sess.run([precision_update_op, recall_update_op], feed_dict={Y_test_placeholder: np.squeeze(Y_test), predictions_placeholder: final_predictions})

        print(precision_value)
        print(recall_value)

        return parameters


# In[90]:


parameters = model(train_x, train_y, test_x, test_y)

