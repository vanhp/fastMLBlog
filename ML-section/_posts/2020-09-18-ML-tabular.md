---
toc: false
layout: post
description: Understanding and handling of structural data
category: ML
title: Working with Tabular and Time-Series Data
image: /ML-section/assets/images/neuralNet1.jpg
hide: false

---
<!-- ![]({{page.image | relative_url}}) -->
{% include alert.html text="Warning: This page is under heavy construction and constant changing" %}

---

Using ML to analyze and make prediction from Data that have structual type such as data that stored in database, CSV file,spreadsheet data, that have table with row and column. Some data also have time sequence such as day, month,holyday...

The objective is to predict the value in one column based on the values in the other columns.

The task is divide into two types:

For structure data types:
1. Ensembles of decision trees (i.e., random forests and gradient boosting machines), mainly for structured data (such as you might find in a database table at most companies). 
    - Most importantly, the critical step of interpreting a model of tabular data is significantly easier for decision tree ensembles than DL.
    - There are tools and methods for answering the pertinent questions, like: Which columns in the dataset were the most important for your predictions? How are they related to the dependent variable? How do they interact with each other? And which particular features were most important for some particular observation?

For unstructure data types:
2. Multilayered neural networks learned with SGD (Deep Learning) (i.e., shallow and/or deep learning), mainly for unstructured data (such as audio, images, and natural language) 
    - DL also work well with structure data type may be slower to train and hard to intepret result than ensemble decision trees which also don't need GPU, less parameter tuning,more mature ecosystem.
    
    - There are some high-cardinality categorical variables that are very important 
      - ("cardinality" refers to the number of discrete levels representing categories, so a high-cardinality categorical variable is something like a zip code, which can take on thousands of possible levels).
    - There are some columns that contain data that would be best understood with a neural network, such as plain text data.

## Tabular data Modeling Deep Dive

There are a few approach available to tackle structural data type.
Such as Decision tree, Random Forest, Neural Net.

Decision tree is one approach that also provide as good result as the same time use less reource than DL which make it suitable for tabular type problem.

### Ensemble Decision Trees
Decision tree don't require matrix multiplication or derivative calculation. So PyTorch is of no help. Scikit-learn is better suit for this task.

Data to use for Decision tree
[bluebookforbulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers/data)

Handling Different data types from tables
Dataset:
the Blue Book for Bulldozers Kaggle competition

```
# read data from csv file
df = pd.read_csv(path/'TrainAndValid.csv', low_memory=False)
df.columns
# inspect data
df['ProductSize'].unique()
sizes = 'Large','Large / Medium','Medium','Small','Mini','Compact'
df['ProductSize'] = df['ProductSize'].astype('category')
df['ProductSize'].cat.set_categories(sizes, ordered=True, inplace=True)
dep_var = 'SalePrice
df[dep_var] = np.log(df[dep_var])

```

The most important data column is the dependent variable since it will be used for prediction.

Another important task is to select the metric. Selecting the appropriate metric is an important part of the project setup that use to gauge the performance of the model.
unless it's already specify the loss function is also must be decided. this is set for root mean square error (RMSLE) between actual and predict value by default.

### Decision Trees
Decision tree works like binary tree where parent node is the middle whith the left is <= parent and right child node larger. Where the leaf node is the prediction

The basic steps to train a decision tree can be written down very easily:

1. Loop through each column of the dataset in turn.
2. For each column, loop through each possible level of that column in turn.
3. Try splitting the data into two groups, based on whether they are greater than or less than that value (or if it is a categorical variable, based on whether they are equal to or not equal to that level of that categorical variable).
4. Find the average sale price for each of those two groups, and see how close that is to the actual sale price of each of the items of equipment in that group. That is, treat this as a very simple "model" where our predictions are simply the average sale price of the item's group.
5. After looping through all of the columns and all the possible levels for each, pick the split point that gave the best predictions using that simple model.
6. We now have two different groups for our data, based on this selected split. Treat each of these as separate datasets, and find the best split for each by going back to step 1 for each group.
7. Continue this process recursively, until you have reached some stopping criterion for each group—for instance, stop splitting a group further when it has only 20 items in it.

### Handling Dates

Since date type have some important in some context but less so in different context, e.g. yesterday, tomorrow,lastweek, holiday, day of week, day of month
fastai provide function add_datepart to do this task

```
df_test = pd.read_csv(path/'Test.csv', low_memory=False)
df_test = add_datepart(df_test, 'saledate')
' '.join(o for o in df.columns if o.startswith('sale'))

```
#### Handling missing data and string

fastai provide `TabularPandas`, and `TabularProc` API which has Categorify, and FillMissing functions to help manage these problems.

TabularProc:

- It returns the exact same object that's passed to it, after modifying the object in place.
- It runs the transform once, when data is first passed in, rather than lazily as the data is accessed.
- Categorify:  replaces a column with a numeric categorical column
- FillMissing: replaces missing values with the median of that column
    - and creates a new Boolean column that is set to True for any row where the value was missing

       ``` procs = [Categorify, FillMissing]```

TabularPandas:

will handle splitting the dataset into training and validation sets

```
cond = (df.saleYear<2011) | (df.saleMonth<10)
train_idx = np.where( cond)[0]
valid_idx = np.where(~cond)[0]

splits = (list(train_idx),list(valid_idx))
```

TabularPandas needs to be told which columns are continuous and which are categorical it can handle that automatically using the helper function `cont_cat_split`

```
cont,cat = cont_cat_split(df, 1, dep_var=dep_var)
to = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)
len(to.train),len(to.valid)
to.show(3)

to1 = TabularPandas(df, procs, ['state', 'ProductGroup', 'Drive_System', 'Enclosure'], [], y_names=dep_var, splits=splits)
to1.show(3)
## check the underline value 
to.items.head(3)

```

### The conversion of categorical data to number

The conversion of categorical columns to numbers is done by simply replacing each unique level with a number
The numbers associated with the levels are chosen consecutively as they are seen in a column, so there's no particular meaning to the numbers in categorical columns after conversion, except the first time, the first convert a column to a Pandas ordered category ordering must be provided.

``` 
to.classes['ProductSize']
(path/'to.pkl').save(to) 
# load back in
to = (path/'to.pkl').load()
xs,y = to.train.xs,to.train.y
valid_xs,valid_y = to.valid.xs,to.valid.y
m = DecisionTreeRegressor(max_leaf_nodes=4)
# train
m.fit(xs, y);
draw_tree(m, xs, size=7, leaves_parallel=True, precision=2)

```

### Categorical Variables

The decision tree can handle these variable with ease since they are treat just like another variables that may group into a node according the splitting criteria in which case it may split down to the leaf node.

### Random Forests

Random forest is a model that comprise with large number of decision trees. It then take the averages of predictions from these decision trees, which are generated by randomly varying various parameters that specify what data is used to train the tree and other tree parameters. 

Random forest uses baggin with randomly choose subset of column making split in each decision tree.
A baggin technique where the data was divide into subset then 

1. randomly pick a subset of the row of data
2. train a model on that subset
3. save the model then repeat from step 1 a few time
4. This process will result in number of trained models them 
5. make prediction from these models then take the average of these prediction


### Create a Random forest

```
def rf(xs, y, n_estimators=40, max_samples=200_000,
       max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,
        max_samples=max_samples, max_features=max_features,
        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)
# n_estimators defines the number of trees we want
# max_samples defines how many rows to sample for training each tree
# max_features defines how many columns to sample at each split point (where 0.5 means "take half the total number of columns")
# min_samples_leaf specify when to stop splitting the tree nodes limiting the depth of the tree
# n_jobs=-1 to tell sklearn to use all our CPU in parallel

m = rf(xs, y);
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)
preds = np.stack([t.predict(valid_xs) for t in m.estimators_])
r_mse(preds.mean(0), valid_y)
plt.plot([r_mse(preds[:i+1].mean(0), valid_y) for i in range(40)]);
r_mse(m.oob_prediction_, y)

```
### Out-of-Bag Error

The OOB error is a way of measuring prediction error on the training set by only including in the calculation of a row's error trees where that row was not included in training. 
 allows us to see whether the model is overfitting, without needing a separate validation set.
 it only use the error from the tree don't use that subset of data for training effectively a validation set

### Model Interpretation

For tabular data, model interpretation is particularly important. For a given model, the things we are most likely to be interested in are:

How confident are we in our predictions using a particular row of data?
For predicting with a particular row of data, what were the most important factors, and how did they influence that prediction?
Which columns are the strongest predictors, which can we ignore?
Which columns are effectively redundant with each other, for purposes of prediction?
How do predictions vary, as we vary these columns?

#### Tree Variance for Prediction Confidence

How confident of the model on the prediction. One simple way is to use the standard deviation of predictions across the trees, instead of just the mean. This tells us the relative confidence of predictions. In general, we would want to be more cautious of using the results for rows where trees give very different results (higher standard deviations), compared to cases where they are more consistent (lower standard deviations).

```
# get the prediction from all trees
preds = np.stack([t.predict(valid_xs) for t in m.estimators_])
preds.shape
preds_std = preds.std(0)
preds_std[:5]

```

#### Feature Importance

want to know how it's making predictions use the attribute `feature_importances_`

```
def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)

def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

plot_fi(fi[:30]);
```

The way these importances are calculated is quite simple yet elegant. The feature importance algorithm loops through each tree, and then recursively explores each branch. At each branch, it looks to see what feature was used for that split, and how much the model improves as a result of that split. The improvement (weighted by the number of rows in that group) is added to the importance score for that feature. This is summed across all branches of all trees, and finally the scores are normalized such that they add to 1.

#### Removing Low-Importance Variables

Since there many variable in the table. Some variable have contribute marginally to the overall prediction. To reduce the comptation and resource these values should be removed.

```
to_keep = fi[fi.imp>0.005].cols
len(to_keep)

xs_imp = xs[to_keep]
valid_xs_imp = valid_xs[to_keep]
m = rf(xs_imp, y)
m_rmse(m, xs_imp, y), m_rmse(m, valid_xs_imp, valid_y)
len(xs.columns), len(xs_imp.columns)
plot_fi(rf_feat_importance(m, xs_imp));
```

#### Removing Redundant Features

By focusing on the most important variables, and removing some redundant ones, greatly simplified our model.
By merging columns that most similar to each other starting from the leaf

```
def get_oob(df):
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=15,
        max_samples=50000, max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(df, y)
    return m.oob_score_
get_oob(xs_imp)
# try removing each of our potentially redundant variables, one at a time:
{c:get_oob(xs_imp.drop(c, axis=1)) for c in (
    'saleYear', 'saleElapsed', 'ProductGroupDesc','ProductGroup',
    'fiModelDesc', 'fiBaseModel',
    'Hydraulics_Flow','Grouser_Tracks', 'Coupler_System')}

to_drop = ['saleYear', 'ProductGroupDesc', 'fiBaseModel', 'Grouser_Tracks']
get_oob(xs_imp.drop(to_drop, axis=1))
xs_final = xs_imp.drop(to_drop, axis=1)
valid_xs_final = valid_xs_imp.drop(to_drop, axis=1)
# save
(path/'xs_final.pkl').save(xs_final)
(path/'valid_xs_final.pkl').save(valid_xs_final)
# load back
xs_final = (path/'xs_final.pkl').load()
valid_xs_final = (path/'valid_xs_final.pkl').load()
m = rf(xs_final, y)
m_rmse(m, xs_final, y), m_rmse(m, valid_xs_final, valid_y)
```

#### Partial Dependence

Try to understand the relationship of the most important predictors

```

p = valid_xs_final['ProductSize'].value_counts(sort=False).plot.barh()
c = to.classes['ProductSize']
plt.yticks(range(len(c)), c);

from sklearn.inspection import plot_partial_dependence
fig,ax = plt.subplots(figsize=(12, 4))
plot_partial_dependence(m, valid_xs_final, ['YearMade','ProductSize'],
                        grid_resolution=20, ax=ax);
```

Data Leakage

Data leakage is subtle and can take many forms. In particular, missing values often represent data leakage.

#### Tree Interpreter

```
!pip install treeinterpreter
!pip install waterfallcharts
#Use water fall to draw a chart
```

to answer this question. For predicting with a particular row of data, what were the most important factors, and how did they influence that prediction?

```
import warnings
warnings.simplefilter('ignore', FutureWarning)

from treeinterpreter import treeinterpreter
from waterfall_chart import plot as waterfall

row = valid_xs_final.iloc[:5]
prediction,bias,contributions = treeinterpreter.predict(m, row.values)
prediction[0], bias[0], contributions[0].sum()
waterfall(valid_xs_final.columns, contributions[0], threshold=0.08, 
          rotation_value=45,formatting='{:,.3f}');
          
```

### Extrapolation and Neural Networks

The value in the dataset determine the minimum and maximum value that Random forest can be. It cannot extrapolate value that beyon this boundary

### Finding Out-of-Domain Data

Sometime the test train, validation dataset may not have the same distribution pattern. Especially, data that don't fit the general norm of rest of dataset. If they are too much different this may skew the result.

#### Using Random forest to help find uneven distribution

If the either the test,validation set have the same distribution with the train set there should be no predicting power another word, it should be 0. Here are step to find out:

1. combine the train and validation 
2. create a dependent variable to reprent the data from each row
3. build the RF
4. check the model feature importance
5. if the value differ significantly this indicate different distribution
6. try to remove this value to see if it affect the outcome

```
df_dom = pd.concat([xs_final, valid_xs_final])
is_valid = np.array([0]*len(xs_final) + [1]*len(valid_xs_final))

m = rf(df_dom, is_valid)
rf_feat_importance(m, df_dom)[:6]

# take the base line of original RMSE value for comparison
m = rf(xs_final, y)
print('orig', m_rmse(m, valid_xs_final, valid_y))

for c in ('SalesID','saleElapsed','MachineID'):
    m = rf(xs_final.drop(c,axis=1), y)
    print(c, m_rmse(m, valid_xs_final.drop(c,axis=1), valid_y))

# look like these 2 variables may be removable with little effect
time_vars = ['SalesID','MachineID']
xs_final_time = xs_final.drop(time_vars, axis=1)
valid_xs_time = valid_xs_final.drop(time_vars, axis=1)

m = rf(xs_final_time, y)
m_rmse(m, valid_xs_time, valid_y)
# tip remove some old data that are less relevance to whole data 
xs['saleYear'].hist();
# let retrain see if it improve!
filt = xs['saleYear']>2004
xs_filt = xs_final_time[filt]
y_filt = y[filt]
m = rf(xs_filt, y_filt)
m_rmse(m, xs_filt, y_filt), m_rmse(m, valid_xs_time, valid_y)

```

### Tabular with Neural Network

In NN categorical data is require the use of embedding.
Fastai can convert categorical variable columns to embedding but it must be specified.
It does this by comparing the number of distinct levels in the variable to the value of the max_card == 9000 parameter.
If it's lower, fastai will treat the variable as categorical else treat as continous variable. It creates embeding very large size but < 10000 is best
```
df_nn = pd.read_csv(path/'TrainAndValid.csv', low_memory=False)
df_nn['ProductSize'] = df_nn['ProductSize'].astype('category')
df_nn['ProductSize'].cat.set_categories(sizes, ordered=True, inplace=True)
df_nn[dep_var] = np.log(df_nn[dep_var])
df_nn = add_datepart(df_nn, 'saledate')
df_nn_final = df_nn[list(xs_final_time.columns) + [dep_var]]

# max_card > 9000 treat as continous else treat as categorical
cont_nn,cat_nn = cont_cat_split(df_nn_final, max_card=9000, dep_var=dep_var)

# use this variable to predict future sale price
# make it a continous variable

#this col has > 9000 but don't want in categorical, move to continous
cont_nn.append('saleElapsed')
# take out from categorical col
cat_nn.remove('saleElapsed')
# check its cardinallity of each variable
df_nn_final[cat_nn].nunique()
# hi # of card mean large # of row in embedding matrix
# remove it but make sure it doesn't affect RF model
xs_filt2 = xs_filt.drop('fiModelDescriptor', axis=1)
valid_xs_time2 = valid_xs_time.drop('fiModelDescriptor', axis=1)
m2 = rf(xs_filt2, y_filt)
m_rmse(m, xs_filt2, y_filt), m_rmse(m2, valid_xs_time2, valid_y)

#look ok, remove it
cat_nn.remove('fiModelDescriptor')

```
Create Panda tabular with normalization (x - mean/std) for NN because care about the scale. However, RF only care about the order of values in the the variable

```
procs_nn = [Categorify, FillMissing, Normalize]
to_nn = TabularPandas(df_nn_final, procs_nn, cat_nn, cont_nn,
                      splits=splits, y_names=dep_var)

```
Can use large batch size since Tabular use less GPU ram than NN.
```
dls = to_nn.dataloaders(1024)
y = to_nn.train.y
# set y_range for regression model
y.min(),y.max()
```
create the learner, set the loss function to MSE
By default, for tabular data fastai creates a neural network with two hidden layers, with 200 and 100 activations, respectively
for larger dataset set them higher

```
from fastai.tabular.all import *
learn = tabular_learner(dls, y_range=(8,12), layers=[500,250],
                        n_out=1, loss_func=F.mse_loss)
learn.lr_find()
learn.fit_one_cycle(5, 1e-2)
preds,targs = learn.get_preds()
r_mse(preds,targs)
learn.save('nn')
```

### Ensemble of Nueral Net and Random Forest get the best of both world?

Since Random Forest itself is comprise with group of Decision trees. 
#### The bagging technique.
It's reasonable to expect that add Neural network into the bag then average the result would be possible. The question is will it improve the performance. 

One problem is that PyTorch use different tensor format e.g. rank-2,scikitlearn/Numpy use rank-1. The squeeze function come in handy to remove the extra axis from PyTorch to Numpy

```
rf_preds = m.predict(valid_xs_time)
ens_preds = (to_np(preds.squeeze()) + rf_preds) /2
r_mse(ens_preds,valid_y)

```
#### The Boosting technique

Instead of averaging the result of the model as in RF.
the step in this approach do:
1. Train a small model that underfits your dataset.
2. Calculate the predictions in the training set for this model.
3. Subtract the predictions from the targets; 
    - these are called the "residuals" and represent the error for each point in the training set.
4. Go back to step 1, but instead of using the original targets, use the residuals as the targets for the training.
5. Continue doing this until you reach some stopping criterion, such as a maximum number of trees, or you observe your validation set error getting worse.

- each new tree will be attempting to fit the error of all of the previous trees combined.
- Because we are continually creating new residuals, by subtracting the predictions of each new tree from the residuals from the previous tree, the residuals will get smaller and smaller.
- To make predictions with an ensemble of boosted trees, we calculate the predictions from each tree, and then add them all together. 


Some well-known models and libraries

- Gradient boosting machines (GBMs) and 
- gradient boosted decision trees (GBDTs) 
- XGBoost

### Combining Embeddings with Other Methods
Embedding (array lookup) may help improve the performance of Neural Network especially at inference time. 



*Note:*

- Entity embedding not only reduces memory usage and speeds up neural networks compared with one-hot encoding, but more importantly by mapping similar values close to each other in the embedding space it reveals the intrinsic properties of the categorical variables... [It] is especially useful for datasets with lots of high cardinality features, where other methods tend to overfit... As entity embedding defines a distance measure for categorical variables it can be used for visualizing categorical data and for data clustering.
- In practice try both methods to see which one better suite for the task
- Deep learning generally work well with highly complex data type
- the leaf node is the predicting result
- non leaf node must be divide further until leaf node
- the criteria for splitting should make sense for the data type 
- check leaf node if there too many of them maybe the result of splitting on unsual condition
- Randdom forest not very sensitive to the hyperparameter choices, such as max_features so the tree can be many as there time to train
- the more tree the more accurate
- set max_samples to default if data is less than 200000 otherwise set to 200000
- use setting for max_features=0.5,min_sample_leaf=4 work well
- generally the first step to improving a model is simplifying it
- Determining Similarity: The most similar pairs are found by calculating the rank correlation, which means that all the values are replaced with their rank (i.e., first, second, third, etc. within the column), and then the correlation is calculated.
- PyTorch unsqueeze() to add axis to data or using x[:,None] in Python the None mean add a axis, to do the same job
- boosting can lead to overfitting with a boosted ensemble, 
    - the more trees you have, the better the training error becomes, and 
    - eventually you will see overfitting on the validation set.


key insign:
1. an embedding layer is exactly equivalent to placing an ordinary linear layer after every one-hot-encoded input layer
2. the embedding transforms the categorical variables into inputs that are both continuous and meaningful.
3. Boosting is in very active research subjects
4. gradient boosted trees are extremely sensitive to the choices of the hyperparameters
    - use a loop that tries a range of different hyperparameters to find the ones that work best.

5. Random forests are the easiest to train, because they are extremely resilient to hyperparameter choices and require very little preprocessing. They are very fast to train, and should not overfit if you have enough trees. But they can be a little less accurate, especially if extrapolation is required, such as predicting future time periods.

Gradient boosting machines in theory are just as fast to train as random forests, but in practice you will have to try lots of different hyperparameters. They can overfit, but they are often a little more accurate than random forests.

Neural networks take the longest time to train, and require extra preprocessing, such as normalization; this normalization needs to be used at inference time as well. They can provide great results and extrapolate well, but only if you are careful with your hyperparameters and take care to avoid overfitting.
6. starting your analysis with a random forest. This will give you a strong baseline you can be confident that it's a reasonable starting point. 
    - Then use that model for feature selection and partial dependence analysis, to get a better understanding of your data.
    - Then try neural nets and GBMs if they give significantly better results on your validation set in a reasonable amount of time, use them. 
    - If decision tree ensembles are working well for you, try adding the embeddings for the categorical variables to the data, and see if that helps your decision trees learn better.

### How fastai treat Tabular Data

fastai's Tabular Classes
In fastai, a tabular model is simply a model that takes columns of continuous or categorical data, and predicts a category (a classification model) or a continuous value (a regression model). Categorical independent variables are passed through an embedding, and concatenated, as we saw in the neural net we used for collaborative filtering, and then continuous variables are concatenated as well.

The model created in tabular_learner is an object of class TabularModel. Take a look at the source for tabular_learner now (remember, that's tabular_learner?? in Jupyter). You'll see that like collab_learner, it first calls get_emb_sz to calculate appropriate embedding sizes (you can override these by using the emb_szs parameter, which is a dictionary containing any column names you want to set sizes for manually), and it sets a few other defaults. Other than that, it just creates the TabularModel, and passes that to TabularLearner (note that TabularLearner is identical to Learner, except for a customized predict method).

That means that really all the work is happening in TabularModel, so take a look at the source for that now. With the exception of the BatchNorm1d and Dropout layers (which we'll be learning about shortly), you now have the knowledge required to understand this whole class. Take a look at the discussion of EmbeddingNN at the end of the last chapter. Recall that it passed n_cont=0 to TabularModel. We now can see why that was: because there are zero continuous variables (in fastai the n_ prefix means "number of," and cont is an abbreviation for "continuous").



### Jagons:
- Continuous variables are numerical data, such as "age," that can be directly fed to the model, 
- Categorical variables contain a number of discrete levels, such as "movie ID," for which addition and multiplication don't have meaning
- bagging process of random take rows of data to train model and make prediction then average over all prediction


