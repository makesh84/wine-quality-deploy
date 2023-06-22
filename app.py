# Import Libraries
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

# Create variable and store flask and pickle file
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# Create a Home route to display the front page


@app.route('/')
def home():
    return render_template('index.html')

# Action will be done when the button is pressed to get prediction from the model


@app.route('/predict', methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    prediction = model.predict(final_features)
    return render_template('index.html', prediction_text="The Wine-Quality is {}".format(prediction[0]))


if __name__ == "__main__":
    app.run(debug=True)
