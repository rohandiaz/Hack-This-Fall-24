import os
import joblib
import nbformat
import pandas as pd
from sklearn.preprocessing import StandardScaler  # Import the scaler

# Set the working directory to the location of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the Jupyter notebook
notebook_filename = 'predict_disease.ipynb'
with open(notebook_filename, 'r', encoding='utf-8') as f:
    notebook_content = nbformat.read(f, as_version=4)

# Execute the code in each cell to load 'best_logistic_model'
for cell in notebook_content.cells:
    if cell.cell_type == 'code':
        exec(cell.source)  # Execute the code in the cell

# Check if 'best_logistic_model' is defined
if 'best_logistic_model' in locals():
    try:
        # Save the model using joblib
        model_filename = os.path.join(os.path.dirname(__file__), "best_logistic_model.pkl")
        joblib.dump(best_logistic_model, model_filename)
        print("Model saved successfully as:", model_filename)
    except Exception as e:
        print("Error saving model:", e)
else:
    print("Error: 'best_logistic_model' is not defined.")

# Assuming you have your features in a DataFrame named 'X' from the notebook
# Create and fit the scaler on the training data
if 'X' in locals():  # Ensure 'X' is defined
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Fit the scaler on your features

    # Save the scaler to a file
    try:
        scaler_filename = os.path.join(os.path.dirname(__file__), "scaler.pkl")
        joblib.dump(scaler, scaler_filename)  # Save the scaler to a file
        print("Scaler saved successfully as:", scaler_filename)
    except Exception as e:
        print("Error saving scaler:", e)
else:
    print("Error: 'X' is not defined. Ensure your features DataFrame is available.")
