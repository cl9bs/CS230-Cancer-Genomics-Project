
# coding: utf-8

# In[1]:


## import library
import gzip
import math
import numpy as np
from sklearn import linear_model, metrics, ensemble
import pandas as pd
import h5py
import tensorflow as tf
import random
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn import svm, datasets
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


##load the data to pandas dataframe
data = pd.read_csv('tcga_RSEM_Hugo_norm_count.gz', compression='gzip', delim_whitespace = True)
print (data.info())


# In[2]:


## create a log file
import logging
import argparse
import os
from utils import Params
from utils import set_logger
from utils import save_dict_to_json

# Set the logger
set_logger(os.path.join('', 'train_various_layers.log'))


# In[ ]:


### Feature Reduction with prior knowledge

##load the cell cycle gene set into a numpy array
select_genes = np.loadtxt('cell_cycle geneset.txt', dtype = np.dtype('str'), skiprows=2) 
print(select_genes.shape)
logging.info("cell_cycle geneset")


# In[ ]:


##load the cell death gene set into a numpy array
select_genes = np.loadtxt('cell_death geneset.txt', dtype = np.dtype('str'), skiprows=2) 
print(select_genes.shape)
logging.info("cell_death geneset")


# In[ ]:


##load the cell adhesion gene set into a numpy array
select_genes = np.loadtxt('cell_adhesion geneset.txt', dtype = np.dtype('str'), skiprows=2) 
print(select_genes.shape)
logging.info("cell_adhesion geneset")


# In[ ]:


##combine the 3 gene sets
select_genes = np.loadtxt('combined geneset.txt', dtype = np.dtype('str'), skiprows=2)
print(select_genes.shape)
logging.info("combined geneset")


# In[ ]:


##select only genes in the cell cycle pathway
data_selected = data.loc[data['sample'].isin(select_genes)]
print (data_selected.shape)
#logging.info("cell cycle gene set selected data shape: " + str(data_selected.shape))


# In[3]:


##clean the data
data_clean = data.select_dtypes(include=['float64'])
column_names = list(data_clean.columns.values)
label = [s[-2:] for s in column_names]
print(len(label))
label = [0 if s in ('01', '02', '03', '04', '05', '06', '07', '08', '09') else 1 for s in label]

print (label.count(0)) #cancer (9807)
print (label.count(1)) #non-cancer (856)

##split the data randomly into training and test
train_index = random.sample(range(0, len(label)), 9000)
train_x = [data_clean.iloc[:,j] for j in train_index]
train_y = [label[j] for j in train_index]
test_x = [data_clean.iloc[:,j] for j in range(0, len(label)) if j not in train_index]
test_y = [label[j] for j in range(0, len(label)) if j not in train_index]


# In[ ]:


### Model 1: Logistic regression with k-fold cross-validation and roc curves
X = data_clean.T
y = np.array(label)

random_state = np.random.RandomState(0)

cv = StratifiedKFold(n_splits = 7)
logreg = linear_model.LogisticRegression(penalty = 'l1', C=1e5, 
                                        random_state = random_state)

f1_scores = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas = logreg.fit(X.iloc[train], y[train]).predict(X.iloc[test])
    print(X.iloc[train].shape, y[train].shape, probas.shape)
    fpr, tpr, thresholds = roc_curve(y[test], probas)
    f1_score = metrics.f1_score(y[test], probas)
    tprs.append(interp(mean_fpr, fpr, tpr))
    f1_scores.append(f1_score)
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw = 1, alpha = 0.3, label = "ROC fold %d (AUC = %0.2f)" % (i, roc_auc))
    i += 1
plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'r', label = 'Luck', alpha = 0.8)
mean_tpr = np.mean(tprs, axis = 0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#plot F1 scores
average = sum(f1_scores)/len(f1_scores)
plt.plot(f1_scores)
plt.ylim(0.0, 1.0)
plt.xlabel('fold')
plt.ylabel('f1_score')
plt.title('F1_score across 7_fold')
plt.axhline(y = average, color = 'r')
plt.show()


# In[ ]:


### Model 2: Random forest
random_forest_reg = ensemble.RandomForestRegressor()
random_forest_reg.fit(train_x, train_y)

##plot the ROC curve for the training set and compute the auc_score
training_set_z_rf = random_forest_reg.predict(test_x)
fpr_rf, tpr_rf, thresholds_rf = metrics.roc_curve(test_y, training_set_z_rf)
training_auc_score_rf = metrics.roc_auc_score(test_y, training_set_z_rf)
plt.plot(fpr_rf, tpr_rf)
plt.title("random_forest_test_roc")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.show()
print("Random Forest Training AUC: " + str(training_auc_score_rf))


# In[ ]:


### Model 3: Support vector machine (SVM)
X = data_clean[0:100].T
y = np.array(label)[1:100]

print(X)
print(Y)

random_state = np.random.RandomState(0)
cv = StratifiedKFold(n_splits = 2)
classifier = svm.SVC(kernel = 'linear', probability = True, 
                    random_state = random_state)

f1_scores = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    print('test')
    probas = classifier.fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])
    print(X.iloc[train].shape, y[train].shape, probas.shape)
    fpr, tpr, thresholds = roc_curve(y[test], probas)
    f1_score = metrics.f1_score(y[test], probas)
    tprs.append(interp(mean_fpr, fpr, tpr))
    f1_scores.append(f1_score)
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw = 1, alpha = 0.3, label = "ROC fold %d (AUC = %0.2f)" % (i, roc_auc))
    i += 1
plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'r', label = 'Luck', alpha = 0.8)
mean_tpr = np.mean(tprs, axis = 0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#plot F1 scores
average = sum(f1_scores)/len(f1_scores)
plt.plot(f1_scores)
plt.ylim(0.0, 1.0)
plt.xlabel('fold')
plt.ylabel('f1_score')
plt.title('F1_score across 7_fold')
plt.axhline(y = average, color = 'r')
plt.show()


# In[4]:


### Model 4: Neural network with TensorFlow 

## turn the data into the right shape
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

train_y = train_y.reshape((train_y.shape[0],1))
test_y = test_y.reshape((test_y.shape[0], 1))

train_x = np.transpose(train_x)
train_y = np.transpose(train_y)
test_x = np.transpose(test_x)
test_y = np.transpose(test_y)


# In[ ]:


#Autoencoder with Keras
m = Sequential()
m.add(Dense(2048,  activation='elu', input_shape=(58581,)))
m.add(Dense(1024,  activation= 'elu'))
m.add(Dense(512,    activation='linear', name="bottleneck"))
m.add(Dense(1024,  activation='elu'))
m.add(Dense(2048,  activation='elu'))
m.add(Dense(58581,  activation='sigmoid'))
m.compile(loss='mean_squared_error', optimizer = Adam())
history = m.fit(train_x, train_x, batch_size=128, epochs=5, verbose=1, validation_data=(test_x, test_x))

encoder = Model(m.input, m.get_layer('bottleneck').output)
Zenc_train = encoder.predict(train_x)  # bottleneck representation
Zenc_test = encoder.predict(test_x)  # bottleneck representation


# In[5]:


##create placeholders
def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an feature vector (58,581 genes)
    n_y -- scalar, number of classes
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    """

    X = tf.placeholder(tf.float32, shape=[n_x, None])
    Y = tf.placeholder(tf.float32, shape=[n_y, None])
    
    return X, Y


# In[6]:


# run placeholder function

X, Y = create_placeholders(58581, 1)
print ("X = " + str(X))
print ("Y = " + str(Y))


# In[7]:


##initialize_parameters
def initialize_parameters(dimensions = [58581, 10, 10, 10, 10, 10, 10, 10, 1]):
    """
    dimensions = [n_x, number of neurons in each layer, n_y]
    Initializes parameters to build a neural network with tensorflow.                       
    
    Returns:
    parameters -- a dictionary of tensors containing Ws and bs
    """
    
    tf.set_random_seed(1)                   # so that our results are reproducible
    parameters = {}
    for i in range(len(dimensions) - 1):
        parameters["W" + str(i+1)] = tf.get_variable("W" + str(i+1), [dimensions[i+1], dimensions[i]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        parameters["b" + str(i+1)] = tf.get_variable("b" + str(i+1), [dimensions[i+1], 1], initializer = tf.zeros_initializer())
    
    logging.info("number of layers: " + str(len(dimensions)-1))
    for j in range (len(dimensions)-1):
        logging.info("number of neurons in " + str(j+1) + " layer: " + str(dimensions[j+1]))

    return parameters


# In[8]:


# run initialize_parameters function
tf.reset_default_graph()
# with tf.Session() as sess:
#     parameters = initialize_parameters([58581, 1000, 500, 500, 250, 1])


# In[9]:


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
    
    A = X
    
    # Retrieve the parameters from the dictionary "parameters" 
    for i in range (int(len(parameters.keys()) / 2)):
        W = parameters["W" + str(i+1)]
        b = parameters["b" + str(i+1)]
        Z = tf.add(tf.matmul(W, A), b)
        A = tf.nn.relu(Z)

    return Z


# In[10]:


# run forward_propagation

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(58581, 1)
    parameters = initialize_parameters([58581, 10, 10, 10, 10, 10, 10, 10, 1])
    Z = forward_propagation(X, parameters)
    print("Z = " + str(Z))


# In[11]:


##compute cost

def compute_cost(Z, Y, pos_weight = 30):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    targets -- vector of labels y (1 or 0) 
    
    
    Returns:
    cost -- runs the session of the cost 
    """
    logging.info("pos_weight: " + str(pos_weight))
    
    # to fit the tensorflow requirement for tf.nn.weighted_cross_entropy_with_logits(...,...)
    logits = Z
    targets = Y
    
    # Use the loss function, pos_weight is set randomly, will tune later
    cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets = targets, logits = logits, pos_weight = pos_weight))
    
    return cost


# In[12]:


# run compute_cost

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(58581, 1)
    parameters = initialize_parameters([58581, 10, 10, 10, 10, 10, 10, 10, 1])
    Z = forward_propagation(X, parameters)
    cost = compute_cost(Y, Z)
    print("cost = " + str(cost))


# In[13]:


# final model
def model(X_train, Y_train, X_test, Y_test, dimensions = [1000, 500, 500, 250], 
          learning_rate = 0.0001, 
          num_epochs = 100, pos_weight = 30, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X_train -- training set, of shape (input size = 20530, number of training examples = 9000)
    Y_train -- test set, of shape (output size = 1, number of training examples = 9000)
    X_test -- training set, of shape (input size = 20530, number of training examples = 1663)
    Y_test -- test set, of shape (output size = 1, number of test examples = 1663)
    learning_rate -- learning rate of the optimization
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    logging.info("learning_rate: " + str(learning_rate))
    logging.info("num_epochs: " + str(num_epochs))
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    

    # Initialize parameters
    parameters = initialize_parameters(dimensions)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z = forward_propagation(X, parameters)
    
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z, Y, pos_weight)
    
    
    # L2 regularization
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1, scope=None)
    weights = tf.trainable_variables() # all vars of your graph
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
    regularized_loss = cost + regularization_penalty # this loss needs to be minimized

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(regularized_loss)
    
    
    # Initialize all the variables
    init = tf.global_variables_initializer()
    
    
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            #epoch_cost = 0.                       # Defines a cost related to an epoch
        
            
            # IMPORTANT: The line that runs the graph on a minibatch.
            # Run the session to execute the "optimizer" and the "cost"
            _ , batch_cost = sess.run((optimizer, cost), feed_dict = {X: X_train, Y: Y_train})

                
            # Print the cost every epoch
            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, batch_cost))
            if print_cost == True and epoch % 10 == 0:
                costs.append(batch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()


        # save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        sigmoid_Z = tf.sigmoid(Z) 
        predictions = tf.cast(tf.map_fn(lambda x: tf.round(x), sigmoid_Z), tf.int32)
        
        
        
        # Precision and recall For training set
        

        Y_train_placeholder = tf.placeholder(tf.int32, shape = Y_train.shape)
        train_predictions_placeholder = tf.placeholder(tf.int32, shape=Y_train.shape)
        train_precision, train_precision_update_op = tf.metrics.precision(labels = Y_train_placeholder, predictions = train_predictions_placeholder)
        
        train_recall, train_recall_update_op = tf.metrics.recall(labels = Y_train_placeholder, predictions = train_predictions_placeholder)
        sess.run(tf.local_variables_initializer())
        train_predictions = sess.run(predictions, feed_dict = {X: X_train})

        train_precision_value, train_recall_value = sess.run([train_precision_update_op, train_recall_update_op], feed_dict={Y_train_placeholder: Y_train, train_predictions_placeholder: train_predictions})

        
        # Precision, recall for test set
        Y_test_placeholder = tf.placeholder(tf.int32, shape=Y_test.shape) 
        predictions_placeholder = tf.placeholder(tf.int32, shape=Y_test.shape)
        precision, precision_update_op = tf.metrics.precision(labels = Y_test_placeholder, predictions = predictions_placeholder)
        
        recall, recall_update_op = tf.metrics.recall(labels = Y_test_placeholder, predictions = predictions_placeholder)
        sess.run(tf.local_variables_initializer())
        final_predictions = sess.run(predictions, feed_dict={X: X_test})
#         print(final_predictions)
#         print(np.squeeze(Y_test))

        precision_value, recall_value = sess.run([precision_update_op, recall_update_op], feed_dict={Y_test_placeholder: Y_test, predictions_placeholder: final_predictions})

        train_F1_score = 2 * train_precision_value * train_recall_value / (train_precision_value + train_recall_value)
        logging.info("train_precision_value: " + str(train_precision_value))
        logging.info("train_recall_value: " + str(train_recall_value))
        logging.info("train_F1_score: " + str(train_F1_score))
        
        F1_score = 2 * precision_value * recall_value / (precision_value + recall_value)
        logging.info("final_cost: " + str(costs[-1]))
        logging.info("precision_value: " + str(precision_value))
        logging.info("recall_value: " + str(recall_value))
        logging.info("F1_score: " + str(F1_score))
        

        return parameters



# In[16]:


## model(train_x, train_y, test_x, test_y, first_layer_neurons, second_layer_neurons,
## learning_rate, num_epochs, pos_weight)
parameters = model(train_x, train_y, test_x, test_y, [58581, 1000, 500, 500, 250, 1], 0.00001, 500, 2)
logging.info('\n')


# In[ ]:


### Integrated Gradients

from deepexplain import DeepExplain

X = tf.placeholder(tf.float32, shape=[58581, None])
tf.reset_default_graph()

# Create a DeepExplain context
with DeepExplain(session = sess) as de:
    parameters = model(train_x, train_y, test_x, test_y, 1000, 500, 0.00001, 100, 16)
    
    def raw_model(x, act=tf.nn.relu):  # < different activation functions lead to different explanations
        layer_1 = act(tf.add(tf.matmul(parameters[0]['W1'], x), parameters[0]['b1']))
        layer_2 = act(tf.add(tf.matmul(parameters[0]['W2'], layer_1), parameters[0]['b2']))
        out_layer = tf.matmul(parameters[0]['W3'], layer_2) + parameters[0]['b3']
        return out_layer

    # Construct model
    logits = raw_model(X)
    
    print(logits)
    
    attributions = {
        'Integrated Gradients': de.explain('intgrad', logits * test_y, X, test_x)
    }
    print('Done')
    print(attributions['Integrated Gradients'])


# In[ ]:


## obtained the index of the most important genes
ig = attributions['Integrated Gradients']
rows, cols = np.where(ig > 0.065)
print(len(set(rows)))  ##5136 IDs with attribution scores greater than 0.05
gene_index = set(rows)
most_relevant_genes = data['sample'][gene_index]
print(most_relevant_genes)
np.savetxt('most_relevant_genes.txt', most_relevant_genes, fmt = '%s')

