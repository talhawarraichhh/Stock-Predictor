from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    # Your prediction logic here
    return f"Prediction result for {ticker}"

if __name__ == '__main__':
    app.run(debug=True)
