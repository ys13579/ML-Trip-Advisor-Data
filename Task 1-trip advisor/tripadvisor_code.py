
#### LOADING THE DATA ####
import pandas as pd
import numpy as np
file = "C:\\Users\\asus\\Downloads\\Data-Science-Intern\\tripadvisor.csv"

data = pd.read_csv(file)
df = pd.DataFrame(data)
df.describe()
list(df)
df.isnull().any()
#### DATA PREPROCESSING ####
df['Helpful votes'].value_counts()


df[['Swimming Pool','Exercise Room', 'Basketball Court', 'Yoga Classes', 'Club', 'Free Wifi']] = df[['Swimming Pool', 'Exercise Room',
 'Basketball Court','Yoga Classes', 'Club', 'Free Wifi']].replace('YES',1 )
df[['Swimming Pool', 'Exercise Room', 'Basketball Court', 'Yoga Classes', 'Club', 'Free Wifi']] = df[['Swimming Pool', 'Exercise Room',
 'Basketball Court', 'Yoga Classes', 'Club', 'Free Wifi']].replace('NO',0 )

#### USER COUNTRY IS SUBSET OF USER CONTINENT SO REMOVING USER CONTINENT  ####
df.rename(columns = {'User country':'Ucountry', 'User continent':'Ucontinent'}, inplace=True)

df.groupby(['Ucountry'])['Ucontinent'].unique()

df.drop('Ucontinent', axis=1, inplace=True)

##### CREATING SEPARATE COLUMNS FOR CATEGORICAL FEATURES  ##### 

one_hot = pd.get_dummies(df[['Period of stay','Traveler type','Ucountry','Review month','Hotel stars','Hotel name']])
df.drop(['Period of stay','Traveler type','Ucountry','Review month','Hotel stars','Hotel name'], axis = 1, inplace=True)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['Review weekday'])
df.drop('Review weekday', axis = 1, inplace=True)
df = df.join(one_hot)

df.drop('Nr. rooms', axis=1, inplace=True)

df['Member years'] = df['Member years'].replace(-1806, 0)
df['Member years'].value_counts()
one_hot = pd.get_dummies(df['Member years'])
df.drop('Member years', axis = 1, inplace=True)
df = df.join(one_hot)

x = df['Score']
df.drop('Score', axis = 1, inplace=True)
df = df.join(x)


#### SHUFFLING THE DATA SINCE CERTAIN FEATURES ARE IN SEQUENCE  ####

from sklearn.utils import shuffle
df = shuffle(df)

from sklearn import preprocessing

minmax_scale = preprocessing.MinMaxScaler().fit(df[['Nr. reviews','Nr. hotel reviews','Helpful votes']])
df[['Nr. reviews','Nr. hotel reviews','Helpful votes']] = minmax_scale.transform(df[['Nr. reviews','Nr. hotel reviews','Helpful votes']])

#### PREDICTIVE MODELLING ####
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

clf1 = svm.SVC(C = 1.0)
clf2 = RandomForestClassifier(random_state=1)
clf3 = LogisticRegression(random_state=1)
clf4 =  KNeighborsClassifier(n_neighbors=3)

X = df.iloc[:, :125]
y = df.iloc[:, 125]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
from sklearn.model_selection import cross_val_score

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# SVM
scores1 = cross_val_score(clf1, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')
print("Cross-validated scores mean:")
print(scores1.mean())

clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)
mean_absolute_percentage_error(y_test, y_pred)

# RANDOM FOREST
scores2 = cross_val_score(clf2, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
print("Cross-validated scores mean:")
print(scores2.mean())

clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
mean_absolute_percentage_error(y_test, y_pred)

#LOGISTIC REGRESSION
scores3 = cross_val_score(clf3, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
print("Cross-validated scores mean:")
print(scores3.mean())

clf3.fit(X_train, y_train)
y_pred = clf3.predict(X_test)
mean_absolute_percentage_error(y_test, y_pred)

#KNEIGHBORS
scores4 = cross_val_score(clf4, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
print("Cross-validated scores mean:")
print(scores4.mean())

clf4.fit(X_train, y_train)
y_pred = clf4.predict(X_test)
mean_absolute_percentage_error(y_test, y_pred)


#### In order to get the best classifier, the splits were made 5 times and the mean of mean absolute percentage error of individual classifiers
#### was taken. 
#### Random forest gave the lowest mean percentage error

#### EXTRACTING IMPORTANT FEATURES ####
feature_importance = clf2.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf2.estimators_],
             axis=0)

# sort the features by its importance
indices = np.argsort(feature_importance)[::-1]

# Print the feature ranking
print("Feature ranking:")

for feature in range(15):
    print("%d. feature %d (%f)" % (feature + 1, indices[feature], feature_importance[indices[feature]]))
    
    
pos = [2,0,1, 64, 115, 114, 80, 63,110,82, 15, 12, 106, 13, 75]
df_cols = df.columns[pos]

#######################################################################################################

