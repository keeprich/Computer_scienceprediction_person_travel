import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Step 1: Prepare the data
dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='B')
prices = np.random.uniform(100, 200, len(dates))
opens = np.random.uniform(100, 200, len(dates))
highs = np.random.uniform(100, 200, len(dates))
lows = np.random.uniform(100, 200, len(dates))
closes = np.random.uniform(100, 200, len(dates))

df = pd.DataFrame({'Date': dates, 'Price': prices, 'Open': opens, 'High': highs, 'Low': lows, 'Close': closes})

# Step 2: Feature engineering
df['Change'] = df['Close'].diff()
df['Profitable'] = np.where(df['Change'] > 0, 1, 0)

# Step 3: Split the data into training and testing sets
X = df[['Open', 'High', 'Low', 'Close']]
y = df['Profitable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Make predictions
predictions = model.predict(X_test)

# Step 6: Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
