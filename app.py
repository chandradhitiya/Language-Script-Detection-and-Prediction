from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

# Load and preprocess the data
data = pd.read_csv("Language_Detection.csv")
X = data["Text"]
y = data["Language"]

le = LabelEncoder()
y = le.fit_transform(y)

# Preprocess the text data
data_list = X.str.replace(r'[!@#$(),n"%^*?:;~`0-9]', ' ').str.replace(r'[[]]', ' ').str.lower()

# Limit the number of features to 5000
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(data_list)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(x_train, y_train)

app = Flask(__name__)

def predict_language(text):
    x = cv.transform([text])
    lang = model.predict(x)
    lang = le.inverse_transform(lang)
    return lang[0]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        user_input = request.form["user_input"]
        prediction = predict_language(user_input)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)
