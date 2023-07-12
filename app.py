import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = RandomForestClassifier()
model.load_model('model.pkl')

# Load the label encoder
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
ohe.categories_ = joblib.load('label_encoder.pkl')

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
    app.run(debug=True)
