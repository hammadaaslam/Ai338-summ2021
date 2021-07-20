![hammad sub proj](https://user-images.githubusercontent.com/86198660/126381640-7bad17dc-0771-4714-8df8-3d8e1698488b.PNG)
```
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

YTrain = trainData.Survived;
trainData.drop('Survived',inplace=True,axis=1)

trainData.drop('Name',inplace=True,axis=1)
trainData.drop('Cabin',inplace=True,axis=1)
trainData.drop('Ticket',inplace=True,axis=1)
testData.drop('Name',inplace=True,axis=1)
testData.drop('Cabin',inplace=True,axis=1)
testData.drop('Ticket',inplace=True,axis=1)

trainData["Sex"] = trainData["Sex"].replace(['female','male'],[0,1])
testData["Sex"] = testData["Sex"].replace(['female','male'],[0,1])
trainData["Embarked"] = trainData["Embarked"].replace(['S','Q','C'],[0,1,2])
testData["Embarked"] = testData["Embarked"].replace(['S','Q','C'],[0,1,2])


trainData.fillna(value=0,inplace=True)

testData.fillna(value=trainData['Age'].mean(),inplace=True)
testData.fillna(value=trainData['Fare'].mean(),inplace=True)


print(YTrain.shape)
print(trainData)
print(testData.shape)

knn_model = KNeighborsClassifier()

knn_model.fit(trainData,YTrain)

predictions = knn_model.predict(testData)

Acc_knn = cross_val_score(knn_model, trainData,YTrain, cv=10, n_jobs = -1)
print(Acc_knn)
print('Accuracy Of KNN Model: ', np.mean(Acc_knn), ' Standard Deviation: ', np.std(Acc_knn))

My_submission = pd.DataFrame({
        "PassengerId": testData["PassengerId"],
        "Survived": predictions
    })
My_submission.to_csv(r'C:\Users\ahsanaslam\Desktop\5th semester (hammad)\untitled14\KNN.csv')
```
