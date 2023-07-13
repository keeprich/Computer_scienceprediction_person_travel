import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Prepare the data
dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='B')
prices = np.random.uniform(100, 200, len(dates))
opens = np.random.uniform(100, 200, len(dates))
highs = np.random.uniform(100, 200, len(dates))
lows = np.random.uniform(100, 200, len(dates))
closes = np.random.uniform(100, 200, len(dates))

df = pd.DataFrame({'Date': dates, 'Price': prices, 'Open': opens, 'High': highs, 'Low': lows, 'Close': closes})

# Feature engineering
df['Change'] = df['Close'].diff()
df['Profitable'] = np.where(df['Change'] > 0, 1, 0)

# Train the model
model = RandomForestClassifier()
model.fit(df[['Open', 'High', 'Low', 'Close']], df['Profitable'])

# Create GUI
def predict_profitability():
    open_price = float(open_entry.get())
    high_price = float(high_entry.get())
    low_price = float(low_entry.get())
    close_price = float(close_entry.get())
    
    # Make a prediction
    prediction = model.predict([[open_price, high_price, low_price, close_price]])
    
    # Convert prediction to human-readable label
    label = 'Profitable' if prediction[0] == 1 else 'Not Profitable'
    
    # Update prediction label
    prediction_label.config(text=f"The prediction is: {label}")

root = tk.Tk()
root.title("Stock Profitability Predictor")

# Create input fields
open_label = tk.Label(root, text="Open Price:")
open_label.pack()
open_entry = tk.Entry(root)
open_entry.pack()

high_label = tk.Label(root, text="High Price:")
high_label.pack()
high_entry = tk.Entry(root)
high_entry.pack()

low_label = tk.Label(root, text="Low Price:")
low_label.pack()
low_entry = tk.Entry(root)
low_entry.pack()

close_label = tk.Label(root, text="Close Price:")
close_label.pack()
close_entry = tk.Entry(root)
close_entry.pack()

# Create predict button
predict_button = tk.Button(root, text="Predict", command=predict_profitability)
predict_button.pack()

# Create prediction label
prediction_label = tk.Label(root, text="")
prediction_label.pack()