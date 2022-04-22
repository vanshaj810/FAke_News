from flask import Flask, request, render_template
import pickle
import numpy as np
import function as f
from sklearn.feature_extraction.text import CountVectorizer

app=Flask(__name__,template_folder='templates')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_form():
  data = request.form['text']
  text=f.process_text(data)
  message= CountVectorizer(analyzer=f.process_text).transform(text).toarray()
  #message = CountVectorizer(analyzer=process_text).fit_transform(data)
  pred = model.predict(message)
  return render_template('after.html', data=pred)
  
if __name__ == "__main__":
    app.run(debug=True)