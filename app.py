from flask import Flask, request
import json
import features_extraction
import numpy as np
import joblib
from urllib.parse import urlparse
import warnings
import requests
import ast

app = Flask(__name__, template_folder='template')
app.secret_key = '1F4453C6EA2C5B454D221285FFFFC'
main_path = app.root_path
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

#The function first sends a GET request to the provided URL using the requests.get method. If the response status code is 200
# (i.e., the website is reachable), the function uses the features_extraction.main method to extract features from the website,
# converts the features to a numpy array, loads a pre-trained classifier model from a joblib file, and predicts whether the website
# is a phishing site or not. If the prediction is -1, the function returns -1, indicating that the website is a phishing site.
# If the prediction is 1, the function returns 1, indicating that the website is legitimate.
def get_prediction_from_url(test_url):
    try:
        response = requests.get(test_url)
        if response.status_code == 200:
            print(f"The website {test_url} is reachable.")
            features_test = features_extraction.main(test_url)
            features_test = np.array(features_test).reshape((1, -1))
            print(features_test)
            clf = joblib.load(main_path + '/classifier/model.pkl')
            pred = clf.predict(features_test)
            if pred == -1:
                return int(-1)
            else:
                return int(1)
        else:
            print(f"The website {test_url} is not reachable (status code: {response.status_code}).")
            return int(-1)
    except requests.exceptions.RequestException as e:
        print(f"The website {test_url} is not reachable: {e}")
        return int(-1)


def get_prediction_from_features(list_):
    features_test = np.array(list_).reshape((1, -1))
    clf = joblib.load(main_path + '/classifier/model.pkl')
    pred = clf.predict(features_test)
    if pred == -1:
        return int(-1)
    else:
        return int(1)


@app.route('/analysis', methods=["POST", "GET"])
def analysis():
    url = request.form['sms']
    prediction = get_prediction_from_url(url)


    if prediction == 1:
        print("\n")
        print("\n The website is safe to browse its legitimate")
        print(" ")
        #print("SAFE")
        x = {"value": 0,"category":"URL ***in message is safe."}
        return json.dumps(x)
    
    elif prediction == -1:
        print("phishing")
        x = {"value": 1, "category": "Phishing URL in message is not safe.Plz don't click"}

        return json.dumps(x)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
