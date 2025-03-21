NLP Assistant (Next-Word Prediction, Meaning Lookup & Sentence Correction)

📌 Project Overview

This is a web-based NLP assistant that provides:

Next-word prediction while typing

Word meaning lookup using WordNet

Sentence correction for grammatical errors

The project is built using Python (Flask), TensorFlow, NLTK, TextBlob, and a web interface (HTML, JavaScript).

🚀 Features

Predict next words using an LSTM-based language model

Find the meaning of a word using WordNet

Correct grammatical mistakes in sentences with TextBlob

Simple and interactive web UI

🏗️ Project Structure

/NLP-Assistant
│-- /static
│   │-- script.js  # JavaScript for frontend interaction
│-- /templates
│   │-- index.html  # Main web page
│-- app.py  # Flask backend
│-- train_model.py  # Model training script
│-- model.h5  # Trained LSTM model
│-- tokenizer.pkl  # Saved tokenizer
│-- requirements.txt
│-- README.md

📥 Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/your-username/NLP-Assistant.git
cd NLP-Assistant

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Train the Model (Optional, if not using provided model)

python train_model.py

4️⃣ Run the Flask Server

python app.py

5️⃣ Open in Browser

Go to: http://127.0.0.1:5000

🔧 How It Works

📝 Next-Word Prediction

Type some text in the input field

Click "Predict"

It suggests the next word based on an LSTM-trained model

🔍 Word Meaning Lookup

Enter a word

Click "Find Meaning"

It retrieves the definition from WordNet

✍ Sentence Correction

Type a grammatically incorrect sentence

Click "Correct"

It suggests the corrected version using TextBlob

📌 Tech Stack

Backend: Flask, TensorFlow, NLTK, TextBlob

Frontend: HTML, JavaScript

Model: LSTM-based next-word prediction

📌 Future Improvements



👨‍💻 Author

Shriram Gupta

⚡ Contributing

Pull requests are welcome! If you’d like to improve something, feel free to contribute.

📜 License

This project is open-source under the MIT License.
