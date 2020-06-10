#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
np.set_printoptions(precision = 4)
import catboost
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# In[3]:


from sklearn.datasets import load_iris
iris = load_iris()
data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
iris = data1
iris.head()


# In[4]:


y = iris.target
X_train, X_test, y_train, y_test = train_test_split(iris, y, test_size=0.2)


# In[5]:


x = X_train.drop("target", axis =1)


# In[9]:


null_value_stats = X_train.isnull().sum(axis=0)
null_value_stats[null_value_stats != 0]


# In[8]:


X_train.head()


# In[17]:


X = X_train.drop('target', axis=1)
y = X_train.target


# In[18]:


print(X.dtypes)


# In[19]:


from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import accuracy_score


# In[20]:


model = CatBoostClassifier(
    custom_loss=['Accuracy'],
    random_seed=42,
    logging_level='Silent'
)


# In[25]:


categorical_features_indices = np.where(X.dtypes != np.float)[0]

model.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_test, y_test),
    plot=True
);


# In[26]:


predictions = model.predict(X_test)
predictions_probs = model.predict_proba(X_test)
print(predictions[:10])
print(predictions_probs[:10])


# In[27]:


model_without_seed = CatBoostClassifier(iterations=10, logging_level='Silent')
model_without_seed.fit(X, y, cat_features=categorical_features_indices)

print('Random seed assigned for this model: {}'.format(model_without_seed.random_seed_))


# In[29]:


params = {
    'iterations': 500,
    'learning_rate': 0.1,
    'eval_metric': 'Accuracy',
    'random_seed': 42,
    'logging_level': 'Silent',
    'use_best_model': False
}
train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
validate_pool = Pool(X_test, y_test, cat_features=categorical_features_indices)


# In[31]:


model = CatBoostClassifier(**params)
model.fit(train_pool, eval_set=validate_pool)

best_model_params = params.copy()
best_model_params.update({
    'use_best_model': True
})
best_model = CatBoostClassifier(**best_model_params)
best_model.fit(train_pool, eval_set=validate_pool);

print('Simple model validation accuracy: {:.4}'.format(
    accuracy_score(y_test, model.predict(X_test))
))
print('')

print('Best model validation accuracy: {:.4}'.format(
    accuracy_score(y_test, best_model.predict(X_test))
))


# In[32]:


get_ipython().run_cell_magic('time', '', 'model = CatBoostClassifier(**params)\nmodel.fit(train_pool, eval_set=validate_pool)')


# In[33]:


get_ipython().run_cell_magic('time', '', "earlystop_params = params.copy()\nearlystop_params.update({\n    'od_type': 'Iter',\n    'od_wait': 40\n})\nearlystop_model = CatBoostClassifier(**earlystop_params)\nearlystop_model.fit(train_pool, eval_set=validate_pool);")


# In[34]:


print('Simple model tree count: {}'.format(model.tree_count_))
print('Simple model validation accuracy: {:.4}'.format(
    accuracy_score(y_test, model.predict(X_test))
))
print('')

print('Early-stopped model tree count: {}'.format(earlystop_model.tree_count_))
print('Early-stopped model validation accuracy: {:.4}'.format(
    accuracy_score(y_test, earlystop_model.predict(X_test))
))

