from flask import Flask, render_template, jsonify, request
import numpy as np 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# load the lstm model 
model = tf.keras.models.load_model("model.h5")

# load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# defining max length based on training 
max_length = 100

def predict_next_word(seed_text, num_words=1):
  for _ in range(num_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_length-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis = -1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
      if(index == predicted):
        output_word = word
        break
    seed_text += " " + output_word

  return seed_text


# define Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    num_words = int(data.get("num_words", 1))
    predicted_text = predict_next_word(text, num_words)
    return jsonify({"predicted_text": predicted_text})

if __name__ == "__main__":
    app.run(debug=True)