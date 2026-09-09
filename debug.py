import joblib

# Load the preprocessor artifact
preprocessor = joblib.load('preprocessor.joblib')

# Extract the exact raw column names it expects
expected_columns = preprocessor.feature_names_in_

print("Your model expects exactly these columns:")
print(list(expected_columns))