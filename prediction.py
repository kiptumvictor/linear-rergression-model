import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
victorset_path = 'RTA Dataset.csv'
victor = pd.read_csv(victorset_path)

# Step 1: Handle Missing Values
victor['Educational_level'] = victor['Educational_level'].fillna(victor['Educational_level'].mode()[0])
victor['Driving_experience'] = victor['Driving_experience'].fillna(victor['Driving_experience'].mode()[0])
victor['Type_of_vehicle'] = victor['Type_of_vehicle'].fillna(victor['Type_of_vehicle'].mode()[0])

# Step 2: Encode Categorical Variables
categorical_features = ['Age_band_of_driver', 'Sex_of_driver', 'Educational_level',
                        'Driving_experience', 'Type_of_vehicle']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ], remainder='passthrough')  # Keep other columns as they are

# Step 3: Prepare target variable (Accident_severity)
severity_mapping = {
    'Slight Injury': 1,
    'Serious Injury': 2,
    'Fatal Injury': 3
}
victor['Accident_severity'] = victor['Accident_severity'].map(severity_mapping)

# Check for any NaN values in the target variable and drop them
if victor['Accident_severity'].isna().sum() > 0:
    victor = victor.dropna(subset=['Accident_severity'])

# Step 4: Split the data into features (X) and target (y)
X = victor[categorical_features]
y = victor['Accident_severity']

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing pipeline
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Create and train the model
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', LinearRegression())])
model_pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = model_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model
model_filename = 'road_accident_severity_model.pkl'
joblib.dump(model_pipeline, model_filename)
print(f"Model saved to {model_filename}")

# Load the model (optional, just to demonstrate loading)
loaded_model_pipeline = joblib.load(model_filename)

# Hypothetical data for prediction
hypothetical_data = pd.DataFrame({
    'Age_band_of_driver': ['18-30'],
    'Sex_of_driver': ['Male'],
    'Educational_level': ['Above high school'],
    'Driving_experience': ['1-2yr'],
    'Type_of_vehicle': ['Automobile']
})

# Predict accident severity
predicted_severity = loaded_model_pipeline.predict(hypothetical_data)
print(f"Predicted accident severity: {predicted_severity[0]}")
