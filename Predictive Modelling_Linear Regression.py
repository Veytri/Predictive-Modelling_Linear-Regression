#!/usr/bin/env python
# coding: utf-8

# Note: Use this template to develop your project. Do not change the steps. For each step, you may add additional cells if needed.

# ## Predictive Modelling with Cancer Information
# 
# <b>Machine Learning & Computational Intelligence</b>
# <b>Training the Weights of A Linear Regression Model</b><br>

# #### Import Libraries
# 
# Importing standard modules; pandas, numply, matplotlib

# In[1]:


import pandas as pd # Import pandas library for data manipulation and analysis
import numpy as np # Import numpy library for array operations
import matplotlib.pyplot as plt # Import pyplot module for creating visualizations
get_ipython().run_line_magic('config', 'Completer.use_jedi=False # Disable the Jedi autocompletion engine')


# #### Load the dataset

# In[2]:


# Read the CSV file into pandas DataFrame
dataset = pd.read_csv('assignment1_dataset.csv')
dataset.head() # Display the first few rows (5) of the dataset


# #### Define the loss function
# 
# Calculate the loss using the mean squared error with the formula: $J_{mse}(y,\hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (y^i - \hat{y}^i)^2$

# In[3]:


# Define mean squared error loss function
def loss_fn(y, yhat):
    loss = (1 / len(y)) * np.sum((yhat - y)**2)
    return loss


# #### Define function to perform prediction
# 
# Since the dataset is a multivariate linear regression, the model is expressed as: 
# $ \hat{y} = \hat{f}(\mathbf{x}) = \sum_{j=0}^{d} w_j x_j$
# where $w_j x_j$ is the inner product between the model's parameters (or weights) and the input features
# 

# In[4]:


# Define predict function
def predict(w, X):
    yhat = np.dot(X, w)
    return yhat


# #### Define function for model training
# The training loss value for each epoch of the training loop is displayed. This function requires four parameters:<br><br>
# $X$ = input features<br>
# $y$ = responses<br>
# alpha = learning rate<br>
# max_epoch = maximum epochs<br>
# 
# A <b>column of ones</b> is added to the input features as the <b>bias/intercept term</b>. It is an additional parameter in linear regression models that allows the model to capture the inherent bias or offset in the data. It represents the expected target value when all the input features are zero.

# In[5]:


# Define the training function
def train_model(X, y, alpha, max_epoch):
    
    num_samples = X.shape[0]
    num_features = X.shape[1]
    
    # Add a column of ones to X for the bias term
    X = np.c_[np.ones(num_samples), X]
    
    # Initialize weights and loss history
    w = np.zeros(num_features + 1)
    hist_loss = []
    
    # Loop until the maximum iteration assigned
    for epoch in range(max_epoch + 1):
        
        # Compute predicted values and loss
        yhat = predict(w, X)
        loss = loss_fn(y, yhat)
        
        # Compute gradient and update weights
        gradient = np.dot(X.T, yhat - y) / num_samples
        w = w - (alpha * gradient)
        
        # Store loss for plotting
        hist_loss.append(loss)
        
        # Display training loss for each epoch
        print(f"Epoch {epoch + 1}/{max_epoch} - Training loss: {loss:.3f}")
        
    return w, hist_loss


# #### Split the dataset
# The training and test is split using the ratio 8:2.

# In[6]:


from sklearn.model_selection import train_test_split

# Assume the target variable is in a column called "target"
X = dataset[['f1','f2','f3','f4','f5']]  # Drop the target column to get the feature matrix
y = dataset['response']  # Get the target vector

# Assume X and y are your feature matrix and target vector, respectively
# Split the dataset into training and testing sets with ratio 8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# #### Train the model

# In[7]:


# Train the model on the training data using the specified learning rate and max epoch
w, hist_loss = train_model(X_train, y_train, alpha=0.003, max_epoch=4000)


# #### Display the estimated weights

# In[8]:


bias_weight = w[0] # Extract the bias weight from the weight vector, w

# Create a DataFrame to store the feature names and their corresponding weights
input_feature_weights = pd.DataFrame(np.array((X_train.columns, w[1:])).T,
columns=['Features', 'Estimated Weights']) 

# Print the weightage of the bias term, w0
print('w0:', bias_weight)


# In[9]:


# Print the weights corresponding to the input features
input_feature_weights


# #### Display the training loss against epoch graph

# In[10]:


# Plot the training loss over epochs
plt.plot(hist_loss)
plt.title("Training Loss vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.show()


# #### Predict the test set

# In[11]:


X_test = np.c_[np.ones(X_test.shape[0]), X_test] # Add a column of ones to X_test for the bias term

y_pred = predict(w, X_test) # Make predictions on the test data

y_pred[0:5] # Display the predicted values


# In[12]:


y_test[0:5] # Compare to the actual values


# #### Display the r2 score, mean squared error and mean absolute error

# In[13]:


# Import evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Evaluate the model using various metrics
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R-Squared Error:', r2_score(y_test, y_pred))


# #### Display the predicted value against actual value graph
# 
# This graph is used to visually evaluate the performance of this regression model. From this graph, we can see that the predicted values align closely with the actual values. This indicates that the model is performing well and accurately predicting the target variable.<br><br>
# The red line is plotted to show that a perfect prediction would all fall on this specific line.

# In[14]:


# Scatter plot of predicted values vs. actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Y: $Y_i$')
plt.ylabel('Predicted Y: $\hat{Y}_i$')
plt.title('Actual vs Predicted Y: $Y_i$ vs $\hat{Y}_i$')

# Plot a line, a perfect prediction would all fall on this line
x = np.linspace(0, 50, 100)
y = x
plt.plot(x, y, 'r')


# In[ ]:




