#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[23]:


#read the file
df = pd.read_csv("F:\indian-liver-patient-records\indian_liver_patient.csv")
df.columns
df.describe(include='all')


# In[24]:


df.head(10)


# In[25]:


plt.rcParams['figure.figsize'] = [30, 20]
for col in df.columns:
    print(col,df[col].unique(), sep = "-")
    


# In[26]:


np.where(pd.isnull(df))


# In[27]:


[df.iloc[i,j] for i,j in zip(*np.where(pd.isnull(df)))]


# In[28]:


df.iloc[:,9].fillna((df.iloc[:,9].mean()), inplace=True)


# In[90]:





# In[29]:


[df.iloc[i,j] for i,j in zip(*np.where(pd.isnull(df)))]


# In[30]:


df.describe(include='all')


# In[31]:


df.corr()


# Total_bilirubin high correlation with Direct_Bilirubin
#Alamine_Aminotransferase high correlation with Aspartate_Aminotransferase
#Total Protein high correlation with Aspartate_Aminotransferase Albumin
#check Albumin and Albumin and Globulin ratio


# In[12]:


import seaborn as sns
g = sns.FacetGrid(df, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Age", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Direct_Bilirubin", "Total_Bilirubin", edgecolor="w")
plt.subplots_adjust(top=0.9)
sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=df, kind="reg")
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Aspartate_Aminotransferase", "Alamine_Aminotransferase",  edgecolor="w")
plt.subplots_adjust(top=0.9)
sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase", data=df, kind="reg")
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Total_Protiens", "Albumin",  edgecolor="w")
plt.subplots_adjust(top=0.9)

sns.jointplot("Total_Protiens", "Albumin", data=df, kind="reg")
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Albumin", "Albumin_and_Globulin_Ratio",  edgecolor="w")
plt.subplots_adjust(top=0.9)
sns.jointplot("Albumin_and_Globulin_Ratio", "Albumin", data=df, kind="reg")


# In[10]:


df.plot(kind='box', subplots=True,  sharex=True, sharey=True)
plt.show()
df.hist()


# In[11]:




scatter_matrix(df)
plt.show()


# In[32]:



df['Gender'] = np.where(df['Gender'] == "Female", 0, 1)


# In[33]:



print(df.groupby('Dataset').size())


# In[63]:





# In[34]:


df.describe()


# In[35]:


df.hist()


# In[39]:


# Total_bilirubin high correlation with Direct_Bilirubin
#Alamine_Aminotransferase high correlation with Aspartate_Aminotransferase
#Total Protein high correlation with Aspartate_Aminotransferase Albumin
#check Albumin and Albumin and Globulin ratio
#df.drop(['Direct_Bilirubin', 'Aspartate_Aminotransferase','Aspartate_Aminotransferase','Albumin'],axis = 1)
df.drop([ 'Aspartate_Aminotransferase','Aspartate_Aminotransferase','Albumin'],1,inplace=True)
df.describe()


# In[71]:


array = df.values
X = array[:,0:6]
Y = array[:,7]
validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

print(X)


# In[72]:


print(Y)


# In[73]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,Y_train)
y_pred=logreg.predict(X_validation)


# In[74]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(Y_validation, y_pred)
cnf_matrix


# In[75]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
Text(0.5,257.44,'Predicted label')


# In[76]:


logreg_score = round(logreg.score(X_train, Y_train) * 100, 2)
logreg_score_test = round(logreg.score(X_validation, Y_validation) * 100, 2)
#Equation coefficient and Intercept
print('Logistic Regression Training Score: \n', logreg_score)
print('Logistic Regression Test Score: \n', logreg_score_test)
print('Coefficient: \n', logreg.coef_)
print('Intercept: \n', logreg.intercept_)
print('Accuracy: \n', accuracy_score(Y_validation,y_pred))
print('Confusion Matrix: \n', confusion_matrix(Y_validation,y_pred))
print('Classification Report: \n', classification_report(Y_validation,y_pred))

sns.heatmap(confusion_matrix(Y_validation,y_pred),annot=True,fmt="d")


# In[77]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
#Predict Output
gauss_predicted = gaussian.predict(X_validation)

gauss_score = round(gaussian.score(X_train, Y_train) * 100, 2)
gauss_score_test = round(gaussian.score(X_validation, Y_validation) * 100, 2)
print('Gaussian Score: \n', gauss_score)
print('Gaussian Test Score: \n', gauss_score_test)
print('Accuracy: \n', accuracy_score(Y_validation, gauss_predicted))
print(confusion_matrix(Y_validation,gauss_predicted))
print(classification_report(Y_validation,gauss_predicted))

sns.heatmap(confusion_matrix(Y_validation,gauss_predicted),annot=True,fmt="d")


# In[78]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
#Predict Output
rf_predicted = random_forest.predict(X_validation)

random_forest_score = round(random_forest.score(X_validation, Y_validation) * 100, 2)
random_forest_score_test = round(random_forest.score(X_validation, Y_validation) * 100, 2)
print('Random Forest Score: \n', random_forest_score)
print('Random Forest Test Score: \n', random_forest_score_test)
print('Accuracy: \n', accuracy_score(Y_validation,rf_predicted))
print(confusion_matrix(Y_validation,rf_predicted))
print(classification_report(Y_validation,rf_predicted))

sns.heatmap(confusion_matrix(Y_validation,rf_predicted),annot=True,fmt="d")


# In[79]:


from sklearn.ensemble import RandomForestClassifier
svm = SVC(gamma='auto')
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[80]:


seed = 7
scoring = 'accuracy'
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('RF',RandomForestClassifier(n_estimators=100)))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# In[81]:


svm = SVC(gamma='auto')
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

sns.heatmap(confusion_matrix(Y_validation,predictions),annot=True,fmt="d")


# In[82]:


LDA = LinearDiscriminantAnalysis()
LDA.fit(X_train, Y_train)
predictions = LDA.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

sns.heatmap(confusion_matrix(Y_validation,predictions),annot=True,fmt="d")


# In[ ]:




