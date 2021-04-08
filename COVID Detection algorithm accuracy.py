#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import sklearn
import pandas as pd

#print('Python: {}'.format(sys.version))
#print('Numpy: {}'.format(numpy.__version__))
#print('Sklearn: {}'.format(sklearn.__version__))
#print('Pandas: {}'.format(pandas.__version__))


# In[2]:


from Bio import SeqIO
'''for i in SeqIO.parse('MN908947_(COVIDSeq).txt', "fasta"):
    print('Id: ' + i.id + '\nSize: ' + str(len(i))+' nucleotides')
'''


# In[3]:


covidSeq = 'MN908947_(COVIDSeq).txt'
with open(covidSeq) as text: 
    print (text.read(100000))


# In[4]:


strand = 'MN908947_(COVIDSeq).txt'
names = ['Sequence']
data = pd.read_csv(strand, names = names)


# In[5]:


print(data.iloc[5])


# In[6]:


covid_class = []
rows, _ = data.shape
for i in range(rows):
    covid_class.append("+")


# In[7]:


covid_id = []
rows,_ = data.shape
for i in range(rows):
    covid_id.append("covid_" + str(i))


# In[8]:


data.insert(0, "Class", covid_class, True)
data.insert(1, "ID", covid_id, True)
data.info()


# In[9]:


sequences = list(data.loc[:, 'Sequence'])
dataset = {}

# loop through sequences and split into individual nucleotides
for i, seq in enumerate(sequences):
    
    # split into nucleotides, remove tab characters
    nucleotides = list(seq.lower())
    nucleotides = [x for x in nucleotides if x != '\t']
    
    nucleotides.append('+')
    
    # add to dataset
    dataset[i] = nucleotides
    #print("Row:" + i, nucleotides)
    #print('Row {}, len {}'.format(i, len(nucleotides)))
    
print(dataset[0])


# In[10]:


# turn dataset into pandas DataFrame if datasets are not the same size, fill remaining with NaN
#df = pd.DataFrame({key: pd.Series(value) for key, value in dataset.items()})

# turn dataset into pandas DataFrame
dframe = pd.DataFrame(dataset)
print(dframe)


# In[11]:


df = dframe.transpose()
print(df.iloc[:5])


# In[12]:


df.rename(columns = {70: 'Class'}, inplace = True) 
print(df.iloc[:5])


# In[13]:


#df.info()
df.describe()


# In[14]:


series = []
for name in df.columns:
    series.append(df[name].value_counts())
    
info = pd.DataFrame(series)
details = info.transpose()
print(details)


# In[15]:


df = numerical_df = pd.get_dummies(df)
numerical_df.iloc[:5]


# In[16]:


df.rename(columns = {'Class_+': 'Class'}, inplace = True)
print(df.iloc[:5])


# In[17]:


from sklearn import model_selection

# Create X and Y datasets for training
X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])

# define seed for reproducibility
seed = 1

# split data into training and testing datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, shuffle = True, random_state=seed)
print("Working")


# In[18]:


print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)
print(np.unique(X_test))
print(np.unique(y_test))
print(np.unique(X_train))
print(np.unique(y_train))


# In[20]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# define scoring method
scoring = 'accuracy'

# Define models to train
names = ["Nearest Neighbors",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "SVM Linear", "SVM RBF", "SVM Sigmoid"]

classifiers = [
    KNeighborsClassifier(n_neighbors = 3),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    SVC(kernel = 'linear'), 
    SVC(kernel = 'rbf'),
    SVC(kernel = 'sigmoid')
]

models = zip(names, classifiers)

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, shuffle = True, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('Test-- ',name,': ',accuracy_score(y_test, predictions))
    print()
    print(classification_report(y_test, predictions))


# In[ ]:




