import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pickle

# Load and preprocess the dataset
df = pd.read_csv("first_telc.csv")

# Define the feature columns and target column
feature_cols = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner', 'Dependents', 'PhoneService',
                'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure']
target_col = 'Churn'  # Assuming the target column is 'Churn'

# Preprocess the tenure column and create bins
# df['tenure_group'] = pd.cut(df.tenure.astype(int), range(1, 80, 12), right=False,
#                             labels=["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)])
# df.drop(columns=['tenure'], axis=1, inplace=True)

# Define the preprocessing for categorical features
categorical_features = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner', 'Dependents', 'PhoneService',
                'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure']
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Split the data
X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
model.fit(X_train, y_train)

# Save the trained model and preprocessor
with open('churnprediction.pkl', 'wb') as file:
    pickle.dump(model, file)
