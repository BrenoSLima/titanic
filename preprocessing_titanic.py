#--Modules--
import pandas as pd

#--Tools--
from sklearn.preprocessing import StandardScaler

def preprocess_data(path='', filename='train.csv', rescale=False):

    train_df = pd.read_csv(path + filename)
    
    #Substituindo os valores contínuos faltantes pela média dos valores de sua coluna
    train_df.fillna(train_df.mean(), inplace=True)
    train_df.fillna('', inplace=True)   #Substituindo os valores em formmatos de string
    
    #Pegar os valores de saída (target)
    y = train_df['Survived'].copy()
    
    #Limpar colunas desnecessárias
    train_df.drop(['Survived', 'PassengerId', 'Name'], axis=1, inplace=True)

    #reescalar os dados    
    if rescale:
        scaler = StandardScaler()
        train_df[['Age', 'Fare']] = scaler.fit_transform(train_df[['Age', 'Fare']])
    
    #Tratar os dados categóricos
    categorical_columns = ['Pclass', 'SibSp', 'Parch', 'Sex', 'Ticket', 'Cabin', 'Embarked']
    X = pd.get_dummies(train_df, columns=categorical_columns) # poderia usar OneHotEncoder com ColumnTransformer

  
    return (X, y)
