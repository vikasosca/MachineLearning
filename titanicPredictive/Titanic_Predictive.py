import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

x = [4, 5, 10, 4, 3, 11, 14 , 8, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

data = list(zip(x, y)) #zip matches points from x and y array and list converts them to a python list
#print(data)
knn = KNeighborsClassifier(n_neighbors=5) #Fit a KNN model on the model using 1 nearest neighbor
knn.fit(data, classes)
#Declare new point
new_x = 8
new_y =21
new_point = [(new_x,new_y)]
prediction = knn.predict(new_point)
#print(prediction)
plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
plt.text(x=new_x-1.7, y=new_y-0.7, s=f"new point, class: {prediction[0]}")
#plt.show()
#print(np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1))
print("***Titanic Survivor predictions *******")
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
'''
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
'''
train = pd.read_csv("C:/Users/Admin/Desktop/AI/python/Datasets/train.csv")
test = pd.read_csv("C:/Users/Admin/Desktop/AI/python/Datasets/test.csv")
# Check the first few rows
#print(train.head()) 
# Fill missing Age with median
train['Age'].fillna(train['Age'].median())
test['Age'].fillna(test['Age'].median())

train['Embarked'].fillna(train['Embarked'].mode()[0] )
test['Embarked'].fillna(test['Embarked'].mode()[0] )
# Fill missing Fare in test set
train['Fare'].fillna(train['Fare'].median())
test['Fare'].fillna(test['Fare'].median())

#Convert sex to numbers
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male':0,'female':1})

train['Embarked'] = train['Embarked'].map({'S': 0,'C': 1,'Q':2})
test['Embarked'] = test['Embarked'].map({'S': 0,'C': 1,'Q':2})
train = train.fillna(train.median(numeric_only=True))
test = test.fillna(test.median(numeric_only=True))
# Fill missing values again just to be safe
X_train = train.fillna(train.median(numeric_only=True))
X_test = test.fillna(test.median(numeric_only=True))
X_train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
X_test['Embarked']  = test['Embarked'].fillna(test['Embarked'].mode()[0])
#Select features & Targets
features =  ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X_train = X_train[features]
y_train = train['Survived']
X_test = X_test[features]
# Create the model
model = LogisticRegression(max_iter=1000) #maximum number of iterations the solver (optimization algorithm)
                            #will run while trying to find the best model coefficients (weights).
# Train it
model.fit(X_train, y_train)
# Predict survival
predictions = model.predict(X_test)
# Prepare submission
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})

from xgboost import XGBClassifier

submission.to_csv("C:/Users/Admin/Desktop/AI/python/Datasets/titanic_submission_LR.csv", index=False)

print("------- Customer Chirning using LogistiRegression----------------")
#WA_Fn-UseC_-Telco-Customer-Churn
df = pd.read_csv("C:/Users/Admin/Desktop/AI/python/Datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")
#print(df.info())
df['Churn'] = df['Churn'].map({'Yes':1,'No':0}) #convert churn to  Yes -, No-0
df.drop('customerID', axis=1)
#'gender', 'tenure', 'MonthlyCharges', 'Contract'
# Fill missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

#print(df['Churn'].isnull().sum())
# One hot encoding to convert string into numeric
df = pd.get_dummies(df)
#print(df.head())
# Split features and target
X = df.drop('Churn',axis=1)
y = df['Churn']
scaler = StandardScaler() # To avoid max iteration reached issue
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=37)
 
# Run the model to train it and make predictions
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#model = LogisticRegression(max_iter=1000)
#model = RandomForestClassifier(class_weight='balanced', random_state=42)
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
#print(predictions)
# Evaluate
'''
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))
'''