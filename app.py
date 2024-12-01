from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the dataset and preprocess if needed
df = pd.read_csv('city_day.csv', na_values='=')
data2 = df.copy()
numeric_columns = data2.select_dtypes(include=np.number).columns.tolist()
data2[numeric_columns] = data2[numeric_columns].fillna(data2[numeric_columns].mean())
data2 = data2.drop(['Date', 'AQI_Bucket'], axis=1)

# Map city names to integers
dist = data2['City']
dist_set = set(dist)
city_dict = {city: i for i, city in enumerate(dist_set)}
data2['City'] = data2['City'].map(city_dict)

# Split features and labels
features = data2[['City', 'PM2.5', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']]
labels = data2['AQI']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=2)

# Train the model
model = RandomForestRegressor(max_depth=10, random_state=0)
model.fit(X_train, y_train)

# Evaluate model on training data
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
print("Training R-squared (R2) score:", train_r2)

# Evaluate model on test data
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
print("Test R-squared (R2) score:", test_r2)

# Save the trained model and city dictionary
joblib.dump(model, 'aqi_prediction_model.pkl')
joblib.dump(city_dict, 'city_dict.pkl')

# Initialize the Flask app
app = Flask(__name__)

# Define a route to render the predict_aqi.html template
@app.route('/')
def home():
    return render_template('predict_aqi.html')

# Define a route for predicting AQI based on user input
@app.route('/predict', methods=['POST'])
def predict_aqi():
    # Load the trained model and city dictionary
    model = joblib.load('aqi_prediction_model.pkl')
    city_dict = joblib.load('city_dict.pkl')
    
    # Get user input from the request
    user_input = request.form
    
    # Prepare the input data
    user_df = pd.DataFrame({
        'City': [user_input['city']],
        'PM2.5': [float(user_input['pm25'])],
        'NO': [float(user_input['no'])],
        'NO2': [float(user_input['no2'])],
        'NOx': [float(user_input['nox'])],
        'NH3': [float(user_input['nh3'])],
        'CO': [float(user_input['co'])],
        'SO2': [float(user_input['so2'])],
        'O3': [float(user_input['o3'])],
        'Benzene': [float(user_input['benzene'])],
        'Toluene': [float(user_input['toluene'])],
        'Xylene': [float(user_input['xylene'])]
    })
    
    # Map city names to integers using the city dictionary
    user_df['City'] = user_df['City'].map(city_dict)
    
    # Predict AQI for user input
    prediction = model.predict(user_df)
    if prediction <= 50 and prediction >= 0:
        print("Good")
        return render_template('predict_aqi.html', prediction=prediction[0], prediction1="Good")
    elif prediction <= 100 and prediction >= 51:
        print("Moderate")
        return render_template('predict_aqi.html', prediction=prediction[0], prediction1="Moderate")
    elif prediction <= 200 and prediction >= 101:
        print("Satisfactory")
        return render_template('predict_aqi.html', prediction=prediction[0], prediction1="Satisfactory")
    elif prediction <= 350 and prediction >= 201:
        print("Poor")
        return render_template('predict_aqi.html', prediction=prediction[0], prediction1="Poor")
    elif prediction <= 500 and prediction >= 351:
        print("Very Poor")
        return render_template('predict_aqi.html', prediction=prediction[0], prediction1="Very Poor")
    else:
        print("Severe")
        return render_template('predict_aqi.html', prediction=prediction[0], prediction1="Poor")
    # Render the predict_aqi.html template with prediction
    

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
