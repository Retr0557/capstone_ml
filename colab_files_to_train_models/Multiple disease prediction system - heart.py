import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

heart_data = pd.read_csv('../dataset/heart.csv')

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train, X_train_prediction)
print('Accuracy on Training data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data : ', test_data_accuracy)

input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
input_data_scaled = scaler.transform(input_data_as_numpy_array)

prediction = model.predict(input_data_scaled)
print(prediction)

if prediction[0] == 0:
    print('The Person does not have Heart Disease')
else:
    print('The Person has Heart Disease')

filename = 'heart_disease_model_scaled.sav'
pickle.dump(model, open(filename, 'wb'))
scaler_filename = 'heart_scaler.sav'
pickle.dump(scaler, open(scaler_filename, 'wb'))

loaded_model = pickle.load(open('heart_disease_model_scaled.sav', 'rb'))
loaded_scaler = pickle.load(open('heart_scaler.sav', 'rb'))

input_data_scaled = loaded_scaler.transform(np.asarray(input_data).reshape(1, -1))
prediction = loaded_model.predict(input_data_scaled)
print(prediction)

for column in X.columns:
    print(column)
