import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Define the routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    location = request.form['location']
    occupation = request.form['occupation']
    salary = int(request.form['salary'])

    # Prepare the data for prediction
    new_data = {
        'age': [age],
        'location': [location],
        'occupation': [occupation],
        'salary': [salary]
    }

    new_df = pd.DataFrame(new_data)

    # Perform one-hot encoding for new data
    new_df_encoded = ohe.transform(new_df)
    new_df_encoded_cols = ohe.get_feature_names_out(categorical_cols)
    new_df_encoded = pd.DataFrame(new_df_encoded, columns=new_df_encoded_cols, index=new_df.index)

    # Make predictions on the new data
    new_predictions = model.predict(new_df_encoded)

    # Convert predictions to human-readable labels
    predicted_labels = ['Yes' if prediction else 'No' for prediction in new_predictions]

    return render_template('result.html', age=age, location=location, occupation=occupation, salary=salary, prediction=predicted_labels[0])

if __name__ == '__main__':
    # Prepare the data for training
    data = {
        'name': ['Rubi', 'Renell', 'Shaine', 'Kellby', 'Con', 'Mame', 'Ivette', 'James', 'Jedediah', 'Reta', 'Mario', 'Nicolea', 'Veronika', 'Anstice', 'Earvin', 'Lidia', 'Shel'],
        'age': [43, 61, 60, 78, 63, 65, 38, 46, 60, 35, 81, 28, 44, 32, 30, 78, 82],
        'location': ['Japan', 'Portugal', 'China', 'China', 'Sweden', 'Russia', 'Sierra Leone', 'Indonesia', 'Russia', 'Afghanistan', 'Venezuela', 'China', 'Indonesia', 'Democratic Republic of the Congo', 'Luxembourg', 'Brazil', 'China'],
        'occupation': ['Estimator', 'Subcontractor', 'Construction Worker', 'Surveyor', 'Project Manager', 'Construction Expeditor', 'Project Manager', 'Construction Worker', 'Construction Manager', 'Construction Manager', 'Architect', 'Construction Foreman', 'Subcontractor', 'Architect', 'Engineer', 'Surveyor', 'Project Manager'],
        'salary': [35841, 89259, 54964, 42957, 32349, 84775, 52219, 44861, 46915, 50384, 88122, 77829, 31959, 48947, 33588, 39178, 46182],
        'love_travel': [True, False, True, True, True, True, True, False, True, False, True, False, True, False, True, True, False]
    }

    df = pd.DataFrame(data)

    # Drop the 'name' column
    df = df.drop('name', axis=1)

    # Prepare the data for training
    X = df.drop('love_travel', axis=1)  # Input features
    y = df['love_travel']  # Target variable

    # Perform one-hot encoding for categorical variables
    categorical_cols = ['location', 'occupation']
    numerical_cols = ['age', 'salary']

    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_encoded = ohe.fit_transform(X[categorical_cols])
    X_encoded_cols = ohe.get_feature_names_out(categorical_cols)
    X_encoded = pd.DataFrame(X_encoded, columns=X_encoded_cols, index=X.index)
    X_encoded[numerical_cols] = X[numerical_cols]

    # Choose and train the model
    model = RandomForestClassifier()
    model.fit(X_encoded, y)

    app.run(debug=True)
