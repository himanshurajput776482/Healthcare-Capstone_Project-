#!/usr/bin/env python
# coding: utf-8

# In[124]:


#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#Reading the dataset & storing it in a variable
health_df=pd.read_csv("health care diabetes.csv")
health_df.head()


# # Project Task: Week 1

# In[4]:


health_df.shape


# In[5]:


health_df.describe()


# In[6]:


#Check for missing values & columns having no variance
health_df.isnull().any()


# In[7]:


health_df.var()


# In[8]:


health_df['Glucose'].value_counts()


# In[9]:


health_df['BloodPressure'].value_counts()


# In[10]:


health_df['SkinThickness'].value_counts()


# In[11]:


health_df['Insulin'].value_counts()


# In[12]:


health_df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=health_df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)
health_df.isnull().any()


# In[13]:


#Working with Missing Values & Imputation
health_df.sample(10)


# In[14]:


health_df['Glucose'].median()


# In[15]:


health_df['BloodPressure'].median()


# In[16]:


health_df['SkinThickness'].median()


# In[17]:


health_df['Insulin'].median()


# In[18]:


health_df['BMI'].median()


# In[19]:


health_df['Glucose']=health_df['Glucose'].fillna(117)
health_df['BloodPressure']=health_df['BloodPressure'].fillna(72)
health_df['SkinThickness']=health_df['SkinThickness'].fillna(29)
health_df['Insulin']=health_df['Insulin'].fillna(125)
health_df['BMI']=health_df['BMI'].fillna(32)

health_df.isnull().any()


# In[125]:


#Visualization to depict distribution of Variables
from scipy.stats import norm
plt.figure(figsize=(10,7))
sns.set_style('whitegrid')

sns.distplot(x=health_df['Glucose'],color='blue',fit=norm)
plt.title("Distribution of Glucose Variable")
plt.show()


# In[126]:


from scipy.stats import norm
plt.figure(figsize=(10,7))
sns.set_style('whitegrid')
sns.distplot(x=health_df['BloodPressure'],color='red',fit=norm)
plt.title("Distribution of BloodPressure Variable")
plt.show()


# In[127]:


from scipy.stats import norm
plt.figure(figsize=(12,6))
sns.set_style('whitegrid')
sns.distplot(x=health_df['Insulin'],color='green',fit=norm)
plt.title("Distribution of Insulin Variable")
plt.show()


# In[128]:


from scipy.stats import norm
plt.figure(figsize=(10,7))
sns.set_style('whitegrid')
sns.distplot(x=health_df['BMI'],color='black',fit=norm)
plt.title("Distribution of BMI Variable")
plt.show()


# In[24]:


#Visual Exploration of Variables Using Histogram for possible values of Outcome(0,1)
health_df.groupby('Outcome').hist(figsize=(14,12),color='navy',edgecolor='yellow')


# In[25]:


#Dtypes of Variables & Outcome
for col,val in health_df.iteritems():
    if(val.dtype=='int64'):
        if(val.nunique()>5):
            print(col," "+'Integer_Type-Continuous')
        else:
            print(col," "+'Integer_Type-Discrete or Categorical')
    elif (val.dtype=='float64'):
        if(val.nunique()>5):
            print(col," "+'Float_Type-Continuous')
        else:
            print(col," "+'Float_Type but Discrete or Categorical')
    else:
        print(col," "+'Object_DType-Categorical')


# In[26]:


#Count of variables for outcome classes
sns.barplot(x='Outcome',y="Glucose",data=health_df,palette='Set2')
plt.title("Effect of Glucose Variable on Outcome count")
plt.show()


# In[27]:


sns.barplot(x='Outcome',y="BloodPressure",data=health_df,palette='Set1')
plt.title("Effect of BloodPressure Variable on Outcome count")
plt.show()


# In[28]:


sns.barplot(x='Outcome',y="SkinThickness",data=health_df,palette='magma')
plt.title("Effect of skinThickness Variable on Outcome count")
plt.show()


# In[29]:


sns.barplot(x='Outcome',y="Insulin",data=health_df,palette='crest')
plt.title("Effect of Insulin Variable on Outcome count")
plt.show()


# In[30]:


sns.barplot(x='Outcome',y="BMI",data=health_df,palette='Dark2')
plt.title("Effect of BMI Variable on Outcome count")
plt.show()


# In[31]:


sns.barplot(x='Outcome',y="Age",data=health_df,palette='Spectral_r');


# In[32]:


sns.barplot(x='Outcome',y='DiabetesPedigreeFunction',data=health_df,palette='Set1');
plt.title("Effect of DiabetesPedigreeFunction Variable on Outcomecount")
plt.show()


# # Project Task: Week 2
# 

# In[33]:


#Data Exploration
sns.countplot(x='Outcome',data=health_df,palette='PuRd');


# In[34]:


health_df['Outcome'].value_counts()


# In[35]:


health_df.shape


# In[36]:


sns.pairplot(health_df,hue='Outcome',palette='tab10',height=2,diag_kind='hist',markers=['o','s']);


# In[37]:


#Correlation Analysis
health_df.corr()


# In[38]:


plt.figure(figsize=(12,7))
sns.heatmap(health_df.corr()[['Outcome']],annot=True, cmap='RdPu_r');


# In[39]:


plt.figure(figsize=(14,7))
sns.heatmap(health_df.corr(),annot=True, cmap='rainbow');


# In[40]:


#Outlier Detection & Removal
health_df.columns


# In[41]:


fig,ax=plt.subplots(3,3,figsize=(18,20))
ax[0,0].boxplot(health_df['Pregnancies'])
ax[0,1].boxplot(health_df['Glucose'])
ax[0,2].boxplot(health_df['BloodPressure'])
ax[1,0].boxplot(health_df['SkinThickness'])
ax[1,1].boxplot(health_df['Insulin'])
ax[1,2].boxplot(health_df['BMI'])
ax[2,0].boxplot(health_df['DiabetesPedigreeFunction'])
ax[2,1].boxplot(health_df['Age'])
ax[0,0].set_title('Pregnancies')
ax[0,1].set_title('Glucose')
ax[0,2].set_title('Blood Pressure')
ax[1,0].set_title('SkinThcikness')
ax[1,1].set_title('Insulin')
ax[1,2].set_title('BMI')
ax[2,0].set_title('DiabetesPedigreeFunction')
ax[2,1].set_title('Age')
plt.tight_layout()
plt.show();


# In[42]:


health_df.describe()


# In[43]:


#Creating a copy of Health_df
health=health_df.copy()
health.shape


# # Project Task: Week 3

# In[44]:


#Check for significant features
import statsmodels.api as sm
features=health.drop(columns='Outcome')
target=health['Outcome']
sm_model=sm.OLS(target,features).fit()
sm_model
sm_model.summary()


# In[45]:


health.corr()['Outcome']*100


# On analysing the p_value of features, we find that features like { Age,SkinThickness & Insulin)
# are statistically insignificant.

# In[46]:


from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(features,target)
ExtraTreesClassifier()
print(model.feature_importances_)


# In[47]:


plt.figure(figsize=(12,4))
ranked_features=pd.Series(model.feature_importances_,index=health.columns[:-1])
ranked_features.nlargest(8).plot(kind='barh')


# In[48]:


health.corr()['Outcome'].sort_values(ascending=True)*100


# On analysing 1)the p_value of features using OLS Statsmodels we find that features like
# { Age,SkinThickness & Insulin) are statistically insignificant where as 2)ExtraTreeClassifier
# predicts features like (BloodPressure,SkinThickness & Insulin) are leastsignificant whereas
# 3) Correlation matrix describes BloodPressure ,SkinThickness ,Pregnancies,Insulin to be
# amongst least correlated the outcome. So combining results from 1) & 2) and using
# correlation value,

# We conclude that features like Insulin,SkinThickness are not significant.

# In[49]:


#Hence, we might not consider these features for model building.
features=features.drop(columns=['Insulin','SkinThickness'])
features.shape


# In[50]:


#Outlier Detection & removal
q1=health_df['DiabetesPedigreeFunction'].quantile(0.25)
q3=health_df['DiabetesPedigreeFunction'].quantile(0.75)
iqr=q3-q1
max=1.5*iqr+ q3
min=q1-1.5*iqr
print(max,min)


# In[51]:


np.where(health_df['DiabetesPedigreeFunction']>1.3)


# In[52]:


np.where(health_df['DiabetesPedigreeFunction']<-0.32)


# In[53]:


np.where(health_df['Pregnancies']>13)


# In[54]:


np.where(health_df['BMI']>60)


# In[55]:


np.where(health_df['BloodPressure']>110)


# In[56]:


np.where(health_df['BloodPressure']<30)


# In[57]:


outliers={4, 12, 39, 45, 58, 147, 187, 228, 243, 259, 308, 330,
370,371, 395, 593, 621, 622, 661,88, 159, 298, 455}
outliers=list(outliers)
outliers


# In[58]:


health=health.drop(outliers,axis=0)
health.shape


# In[59]:


#Handling Class Imbalanced Data & Over_Sampling
health=health.drop(columns=['Insulin','SkinThickness'])
health.shape


# In[60]:


health['Outcome'].value_counts(normalize=True)*100


# In[61]:


health['Outcome'].value_counts()


# It is evident that Outcome has imbalanced data as 253 observations belong to Class 0 (i.e 34%)
# while 489 observations belong to Class 1(i.e 66%)
# Since the size of the Dataset is small, we can consider :

# #1)Over_Sampling Technique to balance the data
# 
# First we will divide the dataset into two parts, one for training & validation and other for
# testing. So, first we will use Train_test_split to divide the dataset & then use SKF for training
# & validation.

# In[62]:


health.shape


# In[63]:


data=health.drop(columns='Outcome')
target=health['Outcome']


# In[64]:


#Splitting the dataset into 1)Train & Validation Set and 2)Test Set
from sklearn.model_selection import train_test_split
x, x_test, y, y_test = train_test_split(data, target, test_size = 0.2,
random_state = 1)
print(x.shape, x_test.shape, y.shape, y_test.shape)


# In[65]:


y.value_counts()


# In[66]:


pip install imblearn


# In[67]:


#Oversampling by SMOTE
from imblearn.over_sampling import SMOTE
smote=SMOTE(sampling_strategy='minority')
x_os,y_os=smote.fit_resample(x,y)
y_os.value_counts()


# In[73]:


#Startified K-Fold CV
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=5)
skf

StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

from sklearn.neighbors import KNeighborsClassifier
model1=KNeighborsClassifier(n_neighbors=5, p=2,metric='minkowski')

from sklearn.linear_model import LogisticRegression
model2=LogisticRegression(solver = 'newton-cg', C = 1000,
penalty='l2')

from sklearn.ensemble import RandomForestClassifier
model3=RandomForestClassifier(n_estimators=50,max_depth=40)

from sklearn.svm import SVC
model4=SVC()

from sklearn.naive_bayes import GaussianNB
model5=GaussianNB(var_smoothing=1e-07)

from xgboost import XGBClassifier
model6=XGBClassifier(base_score=0.25, booster='gbtree',colsample_bylevel=1, colsample_bynode=1,colsample_bytree=1,n_estimators=50)

from sklearn.model_selection import cross_val_score
def get_scores(model):
    score=cross_val_score(model,x_os,y_os,cv=skf)
    mean_score=np.mean(score)
    print('Model: ',model)
    return mean_score

get_scores(model1)


# In[72]:


get_scores(model2)


# In[74]:


get_scores(model3)


# In[75]:


get_scores(model3)


# In[76]:


get_scores(model5)


# In[77]:


get_scores(model6)


# Comparing the performance of KNN Algorithm with five other Algoritms using Stratified KFold
# CV score, KNN is 3rd ranked model in terms of accuracy while Random Forest gives
# the best result & Naive Bayes being the worst performer.

# # Project Task: Week 4

# In[78]:


#Model Building Using KNeighbours Classifier


# In[79]:


from sklearn.neighbors import KNeighborsClassifier
model1=KNeighborsClassifier(n_neighbors=9, p=2,metric='minkowski')
model1.fit(x_os,y_os)
KNeighborsClassifier(n_neighbors=9)
model1.score(x_os,y_os)


# In[80]:


y_pred1=model1.predict(x_test)
model1.score(x_test,y_test)


# In[82]:


from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred1)
cm


# In[83]:


plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test,y_pred1),annot=True,fmt='0.0f');


# In[84]:


print(classification_report(y_test, y_pred1))


# In[85]:


sensitivity=cm[1,1]/(cm[1,1]+cm[1,0])
specificity=cm[0,0]/(cm[0,0]+cm[0,1])
print("Sensitivity:", sensitivity)
print("Specificity:", np.round((specificity),2))


# In[87]:


#ROC Curve & AUC
from sklearn.metrics import roc_curve,auc
y_curve=model1.predict_proba(x_test)[:,1]
FPR,TPR,threshold=roc_curve(y_test,y_curve)
auc=auc(FPR,TPR)
plt.figure(figsize=(4,4),dpi=100)
plt.plot(FPR,TPR,label="KNeighbors(auc= %0.3f)"% auc)
plt.plot([0,1],[0,1],color='orange',linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[91]:


#Model Building Using Random Forest Classifier
model3=RandomForestClassifier(n_estimators=70,max_depth=20,criterion='entropy',min_samples_split=4,
max_leaf_nodes=25,min_samples_leaf=3,max_samples=0.2,max_features=3)
model3.fit(x_os,y_os)
RandomForestClassifier(criterion='entropy', max_depth=20,
max_features=3,
                      max_leaf_nodes=25, max_samples=0.2,
min_samples_leaf=3,
                      min_samples_split=4, n_estimators=70)
model3.score(x_test,y_test)


# In[92]:


model3.score(x_os,y_os)


# In[93]:


y_pred3=model3.predict(x_test)


# In[94]:


#Classification metrics
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test,y_pred3)
cm=confusion_matrix(y_test,y_pred3)
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred3), annot = True, fmt ='0.0f')


# In[95]:


print(classification_report(y_test, y_pred3))


# In[96]:


sensitivity=cm[1,1]/(cm[1,1]+cm[1,0])
specificity=cm[0,0]/(cm[0,0]+cm[0,1])
print("Sensitivity:", sensitivity)
print("Specificity:", np.round((specificity),2))


# In[98]:


#Roc curve & AUC
from sklearn.metrics import roc_curve,auc
y_curve=model3.predict_proba(x_test)[:,1]
FPR,TPR,threshold=roc_curve(y_test,y_curve)
auc_RF=auc(FPR,TPR)
plt.figure(figsize=(4,4),dpi=100)
plt.plot(FPR,TPR,label="RandomForest(auc= %0.3f)"% auc_RF)
plt.plot([0,1],[0,1],color='orange',linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# # Model Building Using Logistic Regression
# 

# In[99]:


from sklearn.linear_model import LogisticRegression
model2=LogisticRegression(solver = 'newton-cg', C = 1000,
penalty='l2')
model2.fit(x_os,y_os)
LogisticRegression(C=1000, solver='newton-cg')
model2.score(x_os,y_os)


# In[100]:


model2.score(x_test,y_test)


# In[101]:


y_pred2=model2.predict(x_test)


# In[102]:


#Classification Metrics
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test,y_pred2)
cm=confusion_matrix(y_test,y_pred2)
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred2), annot = True, fmt =
'0.0f')


# In[103]:


print(classification_report(y_test, y_pred2))


# In[104]:


sensitivity=cm[1,1]/(cm[1,1]+cm[1,0])
specificity=cm[0,0]/(cm[0,0]+cm[0,1])
print("Sensitivity:", np.round((sensitivity),2))
print("Specificity:", np.round((specificity),2))


# In[105]:


#ROC Curve & AUC
from sklearn.metrics import roc_curve,auc
y_curve=model2.predict_proba(x_test)[:,1]
FPR,TPR,threshold=roc_curve(y_test,y_curve)
auc=auc(FPR,TPR)
plt.figure(figsize=(4,4),dpi=100)
plt.plot(FPR,TPR,label="LogisticRegression(auc= %0.3f)"% auc)
plt.plot([0,1],[0,1],color='orange',linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[107]:


#Model Building Using SVM
from sklearn.svm import SVC
model4=SVC(C=10,gamma=0.0001,probability=True)
model4.fit(x_os,y_os)
SVC(C=10, gamma=0.0001, probability=True)
model4.score(x_os,y_os)


# In[108]:


model4.score(x_test,y_test)


# In[109]:


y_pred4=model4.predict(x_test)


# In[110]:


#Classifiaction Metrics
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test,y_pred4)
cm=confusion_matrix(y_test,y_pred4)
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred4), annot = True, fmt ='0.0f')


# In[111]:


print(classification_report(y_test, y_pred4))


# In[112]:


sensitivity=cm[1,1]/(cm[1,1]+cm[1,0])
specificity=cm[0,0]/(cm[0,0]+cm[0,1])
print("Sensitivity:", np.round((sensitivity),2))
print("Specificity:", np.round((specificity),2))


# In[113]:


#Roc Curve & AUC
from sklearn.metrics import roc_curve,auc
y_curve=model4.predict_proba(x_test)[:,1]
FPR,TPR,threshold=roc_curve(y_test,y_curve)
auc=auc(FPR,TPR)
plt.figure(figsize=(4,4),dpi=100)
plt.plot(FPR,TPR,label="SVC(auc= %0.3f)"% auc)
plt.plot([0,1],[0,1],color='orange',linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[116]:


#Model Building Using Naive Bayes
from sklearn.naive_bayes import GaussianNB
model5=GaussianNB(var_smoothing=1e-07)
model5.fit(x_os,y_os)
GaussianNB(var_smoothing=1e-07)
model5.score(x_os,y_os)


# In[118]:


model5.score(x_test,y_test)


# In[119]:


y_pred5=model5.predict(x_test)


# In[120]:


#Classification Metrics
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test,y_pred5)
cm=confusion_matrix(y_test,y_pred5)
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred5), annot = True, fmt =
'0.0f')


# In[121]:


print(classification_report(y_test, y_pred5))


# In[122]:


sensitivity=cm[1,1]/(cm[1,1]+cm[1,0])
specificity=cm[0,0]/(cm[0,0]+cm[0,1])
print("Sensitivity:", np.round((sensitivity),2))
print("Specificity:", np.round((specificity),2))


# In[123]:


#ROC Curve & AUC
from sklearn.metrics import roc_curve,auc
y_curve=model5.predict_proba(x_test)[:,1]
FPR,TPR,threshold=roc_curve(y_test,y_curve)
auc=auc(FPR,TPR)
plt.figure(figsize=(4,4),dpi=100)
plt.plot(FPR,TPR,label="NaiveBayes(auc= %0.3f)"% auc)
plt.plot([0,1],[0,1],color='orange',linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# Comparing the results of all the models taking into account F1-score,precision,accuracy & AUC.
# Considering AUC , Naive Baiyes Tops the list.
# considering F1-score,precision,accuracy, Random Forest Classifier is at the top.
# Here, I choose Random Forest Classifier as the best model.

# In[ ]:




