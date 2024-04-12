import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Name of the file containing the numbers
file_name = "data/numbers.txt"

# Function to load data from a text file
def load_data(file_name):
    with open(file_name, 'r') as file:
        # Split each line by comma and flatten the list of lists into a single list
        data = [int(number) for line in file for number in line.strip().split(',')]
    return np.array(data)

# Function to prepare data in windows of size window_size
def prepare_data(data, window_size):
    X = np.array([data[i:i + window_size] for i in range(len(data) - window_size)])
    y = data[window_size:]
    return X, y

# Function to add the last number to the dataset and update the model
def update_model(last_number, data, window_size=3):
    # Add the last number to the dataset
    updated_data = np.append(data, last_number)

    # Prepare the data with the updated dataset
    X, y = prepare_data(updated_data, window_size)

    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, updated_data

# Function to predict the most probable number after the last entered number
def predict_number(model, data, last_number, window_size=3):
    # Use the last window_size - 1 numbers and the last entered number
    sequence = np.append(data[-(window_size-1):], last_number).reshape(1, -1)
    future_result = model.predict(sequence)[0]
    return future_result

# Load data from the text file
data = load_data(file_name)

# Model initialization
window_size = 3
model, data = update_model(data[0], data, window_size)  # Start with the first data as an example

while True:
    # Ask the user for the last number that came up
    last_number = int(input("Enter the last number that came up on the roulette (or -1 to exit): "))

    # Check if the user wants to exit
    if last_number == -1:
        break

    # Update the model with the new number
    model, data = update_model(last_number, data, window_size)

    # Predict the most probable number after the last entered number
    future_result = predict_number(model, data, last_number, window_size)

    # Show result
    print(f"The most probable number after the number {last_number} is: {future_result}")
