# traffic-predictor

Certainly! To create a traffic predictor application using machine learning, you'll need to include several components such as data pre-processing, model training, and prediction. Below is a complete Python program to forecast traffic congestion using a simple model. This program assumes you have some historical traffic data with features such as time, day, weather, etc., and the target is traffic congestion level.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib  # For saving the model

# Function to load data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("Error: The specified file was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except pd.errors.ParserError:
        print("Error: Error parsing the file.")
    return None

# Function to preprocess data
def preprocess_data(data):
    # Example preprocessing steps
    try:
        # Fill missing values, if any
        data.fillna(method='ffill', inplace=True)

        # Convert categorical variables to dummy/indicator variables
        data = pd.get_dummies(data)

        # Assume 'traffic_congestion' is the column we are trying to predict
        X = data.drop('traffic_congestion', axis=1)
        y = data['traffic_congestion']

        return X, y
    except KeyError as e:
        print(f"Error: Missing expected column in the data - {e}")
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
    return None, None

# Function to train model
def train_model(X, y):
    try:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a random forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict on the test set
        predictions = model.predict(X_test)

        # Calculate and print the mean squared error
        mse = mean_squared_error(y_test, predictions)
        print(f"Model Mean Squared Error: {mse}")

        return model
    except ValueError as e:
        print(f"Error in model training data: {e}")
    except Exception as e:
        print(f"An error occurred during model training: {e}")
    return None

# Function to save the trained model
def save_model(model, file_path):
    try:
        joblib.dump(model, file_path)
        print(f"Model saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")

# Main function
def main():
    data_file = 'traffic_data.csv'  # Replace with your actual file path
    model_file = 'traffic_predictor_model.pkl'

    # Load the data
    data = load_data(data_file)
    if data is None:
        return

    # Preprocess the data
    X, y = preprocess_data(data)
    if X is None or y is None:
        return

    # Train the model
    model = train_model(X, y)
    if model is None:
        return

    # Save the model
    save_model(model, model_file)

if __name__ == '__main__':
    main()
```

### Notes:
- **Data Source**: The code assumes that you have a CSV file named `traffic_data.csv` containing your historical data with features relevant for traffic prediction.
- **Feature Engineering**: Depending on the data, additional feature engineering steps might be needed.
- **Model Selection**: Here, a Random Forest Regressor is used for prediction. You may choose other models as well suited for your data.
- **Handling Errors**: Common errors related to file handling, missing columns, etc., are managed using try-except blocks.
- **Evaluation**: Mean Squared Error (MSE) is used as an evaluation metric; you can add more metrics depending on your needs.
- **Model Persistence**: The trained model is saved with joblib to be used for later predictions without retraining.
- **Data Privacy**: Ensure any personal data in your dataset is handled according to data protection regulations.

Before running this code, make sure to install the necessary Python libraries in your environment:
```shell
pip install pandas scikit-learn joblib
```

Also, replace `'traffic_data.csv'` with the path to your own dataset file. If your dataset structure differs, you'll need to adjust data preprocessing accordingly.