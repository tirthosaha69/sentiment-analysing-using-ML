from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('Model/model.pkl')
vectorizer = joblib.load('Model/vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    tweet = ""

    if request.method == 'POST':
        tweet = request.form['tweet']
        vec = vectorizer.transform([tweet])
        prediction = model.predict(vec)
        sentiment = "Positive Tweet" if prediction[0] == 1 else "Negative Tweet"

    return render_template('index.html', sentiment=sentiment, tweet=tweet)

if __name__ == '__main__':
    app.run(debug=True)
