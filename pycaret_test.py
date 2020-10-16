import pandas as pd

import time

from pycaret.classification import * 
from sklearn.model_selection import cross_val_score

dataframe = pd.read_csv('Desktop/Python/titanic/data/train.csv')

clf = setup(dataframe, target='Survived')
best_models = compare_models(n_select=10) 
tuned_dt = tune_model(best_models[0])

start_time = time.time()
print(f"""
Best pycaret model Model Score: 
Training set: {tuned_dt.score(X_train, y_train):.4f} 
Test set: {tuned_dt.score(X_test, y_test):.4f}
Cross validation: {cross_val_score(tuned_dt, X, y, cv=5).mean():.4f}
Execution Time: {time.time()-start_time:.2f}s\n""")
