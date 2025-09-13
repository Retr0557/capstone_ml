import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

diabetes_dataset = pd.read_csv('../dataset/diabetes.csv')

X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on Training data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on Test data : ', test_data_accuracy)

input_data = (2, 120, 70, 25, 80, 28.0, 0.5, 30)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
input_data_scaled = scaler.transform(input_data_as_numpy_array)

probability = model.predict_proba(input_data_scaled)[0][1]
print(f"Probability of Diabetes: {probability * 100:.2f}%")

if probability >= 0.5:
    print('The person is likely diabetic')
else:
    print('The person is likely not diabetic')

filename = 'diabetes_model_scaled_logreg_prob.sav'
pickle.dump(model, open(filename, 'wb'))
scaler_filename = 'diabetes_scaler.sav'
pickle.dump(scaler, open(scaler_filename, 'wb'))

loaded_model = pickle.load(open('diabetes_model_scaled_logreg_prob.sav', 'rb'))
loaded_scaler = pickle.load(open('diabetes_scaler.sav', 'rb'))

input_data_scaled = loaded_scaler.transform(np.asarray(input_data).reshape(1, -1))
probability = loaded_model.predict_proba(input_data_scaled)[0][1]
print(f"Loaded Model - Probability of Diabetes: {probability * 100:.2f}%")

for column in X.columns:
    print(column)
