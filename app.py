from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)


def analyze_sentiment(user_input):
    model_path = "text-emotion-model"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    res = classifier(user_input)
    return res
    # else:
    #     classifier = pipeline("automatic-speech-recognition", model="jonatasgrosman/wav2vec2-large-xlsr-53-english")
    #     res = classifier(user_input)
    #     return res


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    user_text = request.form.get('text')
    
    if not user_text or not user_text.strip():
        return redirect(url_for('index'))
    
    results = analyze_sentiment(user_text)[0]
    print(results)
    
    return render_template('results.html', 
                         text=user_text, 
                         sentiment=results['label'],
                         score=results['score'])

if __name__ == '__main__':
    app.run(debug=True)