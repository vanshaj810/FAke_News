from flask import Flask, request, render_template
import pickle
import numpy as np
# import function as f
from sklearn.feature_extraction.text import CountVectorizer

app=Flask(__name__,template_folder='templates')
clf= pickle.load(open('model3.pkl', 'rb'))
cv = pickle.load(open("vectorizer3.pickle", "rb"))

@app.route('/')
def home():
  return render_template('main.html')

@app.route('/predict', methods=['POST'])
def get_form():
    message = request.form['text']
    data = [message]
    # from sklearn.feature_extraction.text import CountVectorizer
    # cv = CountVectorizer()
    # cv.fit(data)
    vect = cv.transform(data)
    print(vect)
    pred = clf.predict(vect)
    print(pred)
    return render_template('second.html', data=pred)

  
if __name__ == "__main__":
    app.run(debug=True)