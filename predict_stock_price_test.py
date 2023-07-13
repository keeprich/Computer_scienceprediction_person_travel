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

# Input values for prediction
open_price = float(input("Enter the open price: "))
high_price = float(input("Enter the high price: "))
low_price = float(input("Enter the low price: "))
close_price = float(input("Enter the close price: "))

# Make a prediction
prediction = model.predict([[open_price, high_price, low_price, close_price]])

# Convert prediction to human-readable label
label = 'Profitable' if prediction[0] == 1 else 'Not Profitable'

# Print the prediction
print(f"The prediction is: {label}")
