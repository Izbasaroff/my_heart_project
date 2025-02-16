import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data = pd.read_csv('heart.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))