"""Machine Learning Project"""
"""Using Adults OpenML data base classification on adult income <50K"""
"""Author: Evan Hauk"""

pip install openml

import openml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# List all datasets and their properties
openml.datasets.list_datasets(output_format="dataframe")

# Get dataset by ID
dataset = openml.datasets.get_dataset(179)

"""**Dataset Info**

*   Using the 'adults' data set from 1996 which poles 48,842 entries with 15 attributes. 
*   Goal to classify X with <=50K or >50K total earnings.

Listing of attributes:
**bold text**
>50K, <=50K --> [0, 1]

age: continuous.

workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, 
State-gov, Without-pay, Never-worked.

fnlwgt: continuous.

education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.

education-num: continuous.

marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.

occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.

relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.

race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.

sex: Female, Male.

capital-gain: continuous.

capital-loss: continuous.

hours-per-week: continuous.

native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.


*   Additional Informatio on attribute data can be found here:

**   https://cseweb.ucsd.edu/classes/sp15/cse190-c/reports/sp15/048.pdf



"""

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)
df = pd.DataFrame(X, columns=attribute_names)
df["class"] = y

"""*   X contains all attributes
*   y contains the classification data where it is either <=50k or >50k

*   48842 entriees
*   15 attributes
*goal to classify on column 15
"""

# Drop all rows with NaN values (bad values)
df2=df.dropna()
df2=df.dropna(axis=0)

# Reset index after drop
df2=df.dropna().reset_index(drop=True)

df2 = df

"""-Reduced the number of rows from 48841 down to 45222 values."""

#try with a one-hot data base this will make more columns and more of a sparse array, and give only 0 or 1 values with more columns
df_one_hot = pd.get_dummies(df2[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'class']])

#y is target of class_>50K, so will be 0 if less than 50k and 1 if greater than 50k
y = df_one_hot['class_>50K']

#drop both class columns
#dropping 'workclass_Never-worked' as from analysis it contains all zeros and is therefore not correlated to any other data
df_one_hot = df_one_hot.drop(['class_<=50K', 'class_>50K', 'workclass_Never-worked'],axis=1)

#drop all values that you added to one_hot
#also drop targer class and fnlwgt as it is an arbitrarily calculated variable
X = df2.drop(['fnlwgt', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'class'],axis=1)

#contain all X values as one-hot except for classification
X = X.join(df_one_hot).astype('int')

#create new array with [sum of columns class == 0, sum of column class == 1]
plotArr = np.zeros((103, 2))

"""### **DATA VISUALIZATION**"""

#for each column and row in the DF of X
for i in range(X.columns.size):
  for j in range(len(X)):
    if int(X.iloc[j, i]) > 0:
      if (y[j] > 0):
        #second array position for i'th column
        plotArr[i, 0] += 1
      else:
        #put to first array position for i'th element
        plotArr[i, 1] += 1

#create dataframe with sum of people in each classification
df_sum = pd.DataFrame({">50K":plotArr[:,0],
                   "<=50K":plotArr[:,1]}, 
                  index = [X.columns.tolist()])

# Create unstacked bar
df_sum[5:12].plot(kind="bar", stacked=True)
plt.title("1994 Census Income")
plt.xlabel("Work Class")
plt.ylabel("Count")

df_sum[12:28].plot(kind="bar", stacked=True)
plt.title("1994 Census Income")
plt.xlabel("Education")
plt.ylabel("Count")

df_sum[28:35].plot(kind="bar", stacked=True)
plt.title("1994 Census Income")
plt.xlabel("Marital Status")
plt.ylabel("Count")

df_sum[35:49].plot(kind="bar", stacked=True)
plt.title("1994 Census Income")
plt.xlabel("Occupation")
plt.ylabel("Count")

df_sum[49:55].plot(kind="bar", stacked=True)
plt.title("1994 Census Income")
plt.xlabel("Relationship")
plt.ylabel("Count")

df_sum[55:60].plot(kind="bar", stacked=True)
plt.title("1994 Census Income")
plt.xlabel("Race")
plt.ylabel("Count")

df_sum[60:62].plot(kind="bar", stacked=True)
plt.title("1994 Census Income")
plt.xlabel("Sex")
plt.ylabel("Count")

df_sum[62:104].plot(kind="bar", stacked=True, width=0.5, fontsize=16, figsize=(20,10))
plt.title("1994 Census Income")
plt.xlabel("Native Country")
plt.ylabel("Count")
plt.show

"""### **TRAIN TEST SPLIT**"""

#test train split on X, y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

"""### **DECISION TREE**"""

from sklearn.metrics import accuracy_score
import sklearn.tree as tree
import sklearn.model_selection as ms
clf = tree.DecisionTreeClassifier( max_depth=3 )

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred))

print(tree.export_text(clf))

plt.figure( figsize=(100,50))

_ = tree.plot_tree( clf, 
                    feature_names= X.columns.values,
                    class_names = ['<=50K', '>50K'],
                   filled = True, rounded=True, fontsize = 75)

"""### **PCA - PRINCIPAL COMPONENT ANALYSIS**"""

from sklearn.decomposition import PCA
#using 96% of the variance
pca = PCA(0.96)
pca.fit(X)
X_trans = pca.transform(X)

pca.components_

pca.explained_variance_ratio_

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pca = PCA(n_components=2)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3, p=2)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))

"""### **INITIAL ANALYSIS**

*   Initial PCA set at 0.96 gives an accuracy of 82%
*   Initial Decision Tree gives accuracy of 83%

After removal of the 'workclass_Never-worked' column, no accuracy has changed in the above tests.

After Min/Max Scaling Data:


*   PCA set at 0.96 gives an accuracy of 79% (-3%)
*   Decision Tree gives accuracy of 84% (+1%)

After increasing tree depth: depth = 10

*   Accuracy Increased to 0.8441807326601312

"""

X_arr = X.astype(float)

print(X_arr.columns.tolist())

#check for some correlations in numbers:

corrs = np.corrcoef(X_arr, rowvar=False)

import seaborn as sn

#will make a larger size heat map to view
plt.figure(figsize = (100, 100))

ax = sn.heatmap(corrs)

"""### **SCALING**"""

df_columns = X.columns

#min / max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.to_numpy())
X = pd.DataFrame(X_scaled, columns=df_columns)

#standard scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

"""### **RANDOM FORESTS**"""

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=200, warm_start=True)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

"""Initial accuracy of Random Forest Classifier with n_estimators = 100:  80%

### **ADABOOST**
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets

#create adaboost classifyer object
abc = AdaBoostClassifier(n_estimators=100,
                         learning_rate=1)

#train model
model = abc.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

"""Initial Accuracy of 79% with n_estimators=50 and learning_rate=1

### **GRADIENTBOOST**
"""

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

kf = KFold(n_splits=15,random_state=5,shuffle=True)
for train_index,val_index in kf.split(X):
    X_train,X_val = X.iloc[train_index],X.iloc[val_index],
    y_train,y_val = y.iloc[train_index],y.iloc[val_index]

gradient_booster = GradientBoostingClassifier(learning_rate=0.5)
gradient_booster.get_params()

gradient_booster.fit(X_train,y_train)
print(classification_report(y_val,gradient_booster.predict(X_val)))

"""accuracy = 85%
n_splits=5,random_state=42,shuffle=True

### **MLP - MULTI-LAYER-PERCEPTRON**
"""

#use regular test train split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
est = MLPClassifier(hidden_layer_sizes=(10,10), 
                   activation='relu',
                   learning_rate_init=0.001,
                   alpha=10
                   ,solver='lbfgs'
                   ,max_iter=1000,
                   warm_start=False)

est.fit(X_train, y_train)
y_pred = est.predict(X_test)
print(accuracy_score(y_test,y_pred))

"""
hidden_layer_sizes=(100), activation="identity" with initial accuracy score of 85%

hidden_layer_sizes=(5,10), 
                   activation='relu',
                   learning_rate_init=0.01,
                   alpha=10
                   ,solver='lbfgs' #'sgd', 'adam'
                   ,max_iter=1000,
                   warm_start=False)

                   Accuracy 85.3"""

print(est.coefs_)
print(est.intercepts_)
