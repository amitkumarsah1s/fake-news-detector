from firebase_functions import https_fn
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import string
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

model = pickle.load(open(MODEL_PATH, 'rb'))
vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))

def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def handle_prediction():
    try:
        data = request.get_json(silent=True) or {}
        raw_text = data.get('text', '')
        
        if not raw_text:
            return jsonify({'error': 'No text provided'}), 400
            
        cleaned_text = wordopt(raw_text)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        
        result = "Fake News" if prediction == 0 else "Not Fake News"
        return jsonify({'prediction': result, 'class': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
@app.route('/', methods=['POST'])
def predict():
    return handle_prediction()

@https_fn.on_request()
def api(req: https_fn.Request) -> https_fn.Response:
    with app.request_context(req.environ):
        return app.full_dispatch_request()

if __name__ == '__main__':
    app.run(port=5001, debug=True)