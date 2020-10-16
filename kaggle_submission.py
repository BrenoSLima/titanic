#--Modules--
import pandas as pd

#--Model--
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('Desktop/Python/titanic/data/train.csv')
y_train = train['Survived'].copy()
train.drop(['Survived'], axis=1, inplace=True)

test = pd.read_csv('Desktop/Python/titanic/data/test.csv')

dataset = pd.concat([train, test])

dataset.fillna(dataset.mean(), inplace=True)
dataset.fillna('', inplace=True)

passengerId = dataset['PassengerId']
dataset.drop(['PassengerId', 'Name'], axis=1, inplace=True)

categorical_columns = ['Pclass', 'SibSp', 'Parch', 'Sex', 'Ticket', 'Cabin', 'Embarked']
dataset = pd.get_dummies(dataset, columns=categorical_columns)

X_train = dataset.iloc[0:len(train),]
X_test = dataset.iloc[len(train):,]

tree = RandomForestClassifier(n_estimators=100, max_depth=10)
tree.fit(X_train, y_train)

y_test = tree.predict(X_test)
print(passengerId.iloc[len(train):,])
submission = pd.DataFrame({'PassengerId': passengerId.iloc[len(train):,], 'Survived': y_test})
submission.to_csv('Desktop/Python/titanic/submission.csv', index=False)
