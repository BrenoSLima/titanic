#--Modules--
import time

#--Model--
from sklearn.neural_network import MLPClassifier

#--Tools--
from preprocessing_titanic import preprocess_data
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


X, y = preprocess_data('Desktop/Python/titanic/data/', 'train.csv', rescale=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

mlp = MLPClassifier(hidden_layer_sizes=[10], max_iter=500)
mlp.fit(X_train, y_train)

start_time = time.time()
print(f"""
Multi Layer Perceptron Model Score: 
Training set: {mlp.score(X_train, y_train):.4f} 
Test set: {mlp.score(X_test, y_test):.4f}
Cross validation: {cross_val_score(mlp, X, y, cv=5).mean():.4f}
Execution Time: {time.time()-start_time:.2f}s\n""")