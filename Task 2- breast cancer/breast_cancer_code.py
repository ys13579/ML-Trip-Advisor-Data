

###### IMPORTING THE DATA  ######
import pandas as pd

file = "C:\\Users\\asus\\Downloads\\Data-Science-Intern\\breast_cancer.csv"

data = pd.read_csv(file)
df = pd.DataFrame(data)

list(df)

########  DATA CLEANING  ########

#### DROPING AND RENAMING COLUMNS ####
df.drop([' 4 for malignant)','Unnamed: 12','Unnamed: 13','Unnamed: 14','Unnamed: 15'], axis=1, inplace=True)
df.columns = ['SCN', 'C_Thickness','U_Cell_Size','U_Cell_Shape','MA','SE_CellSize','BNuclei','BChromatin','NNucleoli','Mitoses','Class']

#### HANDLING MISSING VALUES  ####

df['BNuclei'].value_counts()
df['BNuclei']= df['BNuclei'].replace('?','1' )
df.isnull().any()

####  HANDLING CATEGORICAL DATA  ####
one_hot = pd.get_dummies(df['Class'])
df = df.join(one_hot)

df = df.rename(columns={2: 'benign', 4: 'malignant'})
df.drop('Class', axis=1, inplace=True)

df.drop('SCN', axis=1,inplace=True)



#### NORMALISATION OF THE DATA  ####

from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler()

X= df.iloc[:, :9]
y = df.iloc[:, 9]

X = min_max.fit_transform(X)

#### PREDICTIVE MODELLING  ####


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 =  KNeighborsClassifier(n_neighbors=3)
clf4 = svm.SVC(kernel='linear', C = 1.0)

from sklearn.model_selection import cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


# logistic regression
scores1 = cross_val_score(clf1, X_train, y_train, cv=6)
print("Cross-validated scores mean:")
print(scores1.mean())

clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)
confusion_matrix(y_test, y_pred)
clf1.score(X_test, y_test)

#random forest classifier
scores2 = cross_val_score(clf2, X, y, cv=6)
print("Cross-validated scores mean:")
print(scores2.mean())

clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
confusion_matrix(y_test, y_pred)
clf2.score(X_test, y_test)

# kneighbors classifier
scores3 = cross_val_score(clf3, X, y, cv=6)
print("Cross-validated scores mean:")
print(scores3.mean())

clf3.fit(X_train, y_train)
y_pred = clf3.predict(X_test)
confusion_matrix(y_test, y_pred)
clf3.score(X_test, y_test)

# linear svc
scores4 = cross_val_score(clf4, X, y, cv=6)
print("Cross-validated scores mean:")
print(scores4.mean())

clf4.fit(X_train, y_train)
y_pred = clf4.predict(X_test)
confusion_matrix(y_test, y_pred)
clf4.score(X_test, y_test)




###### ROC CURVE FOR SCORING  #######


from sklearn import metrics
import matplotlib.pyplot as plt

plt.figure()
models = [
{
    'label': 'SVM',
    'model': svm.SVC(kernel='linear', C = 1.0, probability=True),
},{
    'label': 'LR',
    'model': clf1,
},{
    'label': 'RF',
    'model': clf2,
},{
    'label': 'KNN',
    'model': clf3,
}
]


for m in models:
    model = m['model'] 
    model.fit(X_train, y_train) 
    y_pred=model.predict(X_test) 
    # Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1])
    #Calculate Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_test,model.predict(X_test))
    #  plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


###Choosing Random Forest Model since it gave better accuracy on test data and on ROC curve####
####### OPTIMIZATION OF RANDOM FOREST MODEL  #######

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import auc, make_scorer, recall_score, accuracy_score, precision_score


param_grid = { 
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
}

######## Decreasing false positives == Increasing precision score  ##########
### So keeping scoring parameter as precision score and applying GridSearchCV ###

def grid_search_wrapper(refit_score='precision_score'):
   
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(clf2, param_grid, scoring=scorers, refit=refit_score,
                           cv=skf, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # make the predictions
    y_pred = grid_search.predict(X_test)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search

grid_search_clf = grid_search_wrapper(refit_score='precision_score')


#######################################################################################################








