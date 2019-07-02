import pandas as pd
desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',12)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train=pd.read_csv('C:\\Users\\hardi\\Desktop\\titanic1\\train.csv')
test=pd.read_csv('C:\\Users\\hardi\\Desktop\\titanic1\\test.csv')
gender_sub=pd.read_csv('C:\\Users\\hardi\\Desktop\\titanic1\\gender_submission.csv')
plt.grid(True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


def input_age(cols):
    age=cols[0]
    Pclass=cols[1]
    if pd.isnull(age):
       if Pclass==1:
          return 37
       elif Pclass==2:
           return 29
       else:
           return 24
    else:
       return age

train['Age']=train[['Age','Pclass']].apply(input_age,axis=1)
test['Age']=test[['Age','Pclass']].apply(input_age,axis=1)
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#plt.show()
test.drop('Cabin',axis=1,inplace=True)
train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)
test.fillna(value=0,inplace=True)


sex=pd.get_dummies(train['Sex'],drop_first=True)
embark=pd.get_dummies(train['Embarked'],drop_first=True)
train=pd.concat([train,sex,embark],axis=1)

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

sex=pd.get_dummies(test['Sex'],drop_first=True)
embark=pd.get_dummies(test['Embarked'],drop_first=True)
test=pd.concat([test,sex,embark],axis=1)

train.drop('PassengerId',axis=1,inplace=True)
#print(train.head())
x_train=train.drop('Survived',axis=1)
y_train=train['Survived']

x_test=test.drop(['Name','Sex','Embarked','Ticket','PassengerId'],axis=1)
#y_test=pd.DataFrame(gender_sub)


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
predictions=model.predict(x_test)
#print(predictions)

submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})
#filename=titanic_survival_sub.csv
#submission.to_csv(titanic_survival_sub.csv,index=False)
#print('Saved File:'+filename)
filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
