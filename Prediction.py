import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import tensorflow as tf

dataframe = pd.read_csv("diabetes.csv")
dataframe.head()
df_label = dataframe['Outcome']
df_features = dataframe.drop('Outcome', 1)
df_features.replace('?', -99999, inplace=True)
print(df_label.head())
df_features.head()
label = []
for lab in df_label:
    if lab == 1:
        label.append([1, 0])  # class 1
    elif lab == 0:
        label.append([0, 1])  # class 0
        data = np.array(df_features)
label = np.array(label)
print(data.shape,label.shape)
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
x_train.shape
model = Sequential()
model.add(Dense(500, input_dim=8, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=1000, batch_size=70, validation_data=(x_test, y_test))
feature_try = np.array([x_train[0]])
feature_try2 = np.array([x_train[1]])



result =model.predict_classes(feature_try2)

if result==0:
    print("NO Diabetes")
else:
    print("Diabetes")