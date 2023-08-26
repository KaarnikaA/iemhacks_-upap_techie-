import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv(r"E:\Projects\Cancer_prediction\cancer.csv")

# Rename the diagnosis column to 'Label'
df = df.rename(columns={'diagnosis': 'Label'})

# Encode the labels using LabelEncoder
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Extract features (X) and labels (y)
X = df.iloc[:, 2:31].values
y = df['Label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the Logistic Regression model
logistic_model = LogisticRegression(random_state=0)
logistic_model.fit(X_train, y_train)

# Save the trained model to a file
with open("logistic.pkl", 'wb') as model_file:
    pickle.dump(logistic_model, model_file)
