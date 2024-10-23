import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
victorset_path = 'RTA Dataset.csv'
victor = pd.read_csv(victorset_path)

# Handle Missing Values
victor['Educational_level'] = victor['Educational_level'].fillna(victor['Educational_level'].mode()[0])
victor['Driving_experience'] = victor['Driving_experience'].fillna(victor['Driving_experience'].mode()[0])
victor['Type_of_vehicle'] = victor['Type_of_vehicle'].fillna(victor['Type_of_vehicle'].mode()[0])

# Encode Categorical Variables
categorical_features = ['Age_band_of_driver', 'Sex_of_driver', 'Educational_level',
                        'Driving_experience', 'Type_of_vehicle']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ], remainder='passthrough')

# Prepare target variable (Accident_severity)
severity_mapping = {
    'Slight Injury': 1,
    'Serious Injury': 2,
    'Fatal Injury': 3
}
victor['Accident_severity'] = victor['Accident_severity'].map(severity_mapping)

# Drop irrelevant or missing data columns
victor_cleaned = victor.dropna()

# Split the data into features (X) and target (y)
X = victor_cleaned[['Age_band_of_driver', 'Sex_of_driver', 'Educational_level',
                    'Driving_experience', 'Type_of_vehicle']]
y = victor_cleaned['Accident_severity']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing pipeline
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Create a pipeline to preprocess data and apply linear regression
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', LinearRegression())])

# Train the model
model_pipeline.fit(X_train, y_train)

# Save the model for future use
model_filename = 'road_accident_severity_model.pkl'
joblib.dump(model_pipeline, model_filename)
print(f"Model saved to {model_filename}")
