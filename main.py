from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import numpy as np  

app = Flask(__name__)

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

contractions = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

def contr_words(text, contractions_dict):
    text = str(text)
    for contraction, expansion in contractions_dict.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)
    return text

def remove_stopwords(text):
    tokens = nltk.word_tokenize(text.lower())  
    english_stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens_wo_stopwords = [word for word in tokens if word not in english_stopwords] 
    return ' '.join(tokens_wo_stopwords)

def text_to_word2vec_vector(text):
    words = text.split()
    
    vector = np.zeros(model.vector_size)  
    word_count = 0
    
    for word in words:
        if word in model.key_to_index:
            vector += model.get_vector(word)
            word_count += 1
        
    if word_count > 0:
        vector /= word_count
    
    return vector

svm_model = joblib.load('sentiment_svm_model.joblib')
rf_model = joblib.load('sentiment_rf_model.joblib')
lr_model = joblib.load('sentiment_lr_model.joblib')
nb_model = joblib.load('sentiment_nb_model.joblib')

sentiment_labels = {2: 'Positive', 1: 'Neutral', 0: 'Negative'}

def preprocess_text(input_text):
    processed_text = contr_words(input_text.lower(), contractions)
    processed_text = remove_stopwords(processed_text)
    processed_text = re.sub(r'http\S+|www\.\S+', ' ', processed_text)
    processed_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', processed_text)
    return processed_text

def predict_sentiment(text, model):
    processed_text = preprocess_text(text)
    input_vector = text_to_word2vec_vector(processed_text)
    prediction = model.predict([input_vector])[0]
    return sentiment_labels[prediction]

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     svm_result = rf_result = lr_result = nb_result = None
#     input_text = ''
    
#     if request.method == 'POST':
#         input_text = request.form['text']
        
#         svm_result = predict_sentiment(input_text, svm_model)
#         rf_result = predict_sentiment(input_text, rf_model)
#         lr_result = predict_sentiment(input_text, lr_model)
#         nb_result = predict_sentiment(input_text, nb_model)
    
#     return render_template('index.html', 
#                            input_text=input_text,
#                            svm_result=svm_result, 
#                            rf_result=rf_result,
#                            lr_result=lr_result,
#                            nb_result=nb_result)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     result = None
#     input_text = ''
    
#     if request.method == 'POST':
#         input_text = request.form['text']
#         selected_model = request.form['model']
        
#         result = {}
#         if selected_model == 'svm' or selected_model == 'all':
#             result['svm'] = predict_sentiment(input_text, svm_model)
#         if selected_model == 'rf' or selected_model == 'all':
#             result['rf'] = predict_sentiment(input_text, rf_model)
#         if selected_model == 'lr' or selected_model == 'all':
#             result['lr'] = predict_sentiment(input_text, lr_model)
#         if selected_model == 'nb' or selected_model == 'all':
#             result['nb'] = predict_sentiment(input_text, nb_model)
    
#     return render_template('index.html', 
#                            input_text=input_text,
#                            result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    input_text = ''
    selected_model = ''

    model_accuracies = {
        'rf': 0.76,
        'lr': 0.76,
        'svm': 0.79,
        'nb': 0.67
    }
    
    if request.method == 'POST':
        input_text = request.form['text']
        selected_model = request.form['model']
        
        result = {}
        if selected_model == 'svm' or selected_model == 'all':
            result['svm'] = predict_sentiment(input_text, svm_model)
        if selected_model == 'rf' or selected_model == 'all':
            result['rf'] = predict_sentiment(input_text, rf_model)
        if selected_model == 'lr' or selected_model == 'all':
            result['lr'] = predict_sentiment(input_text, lr_model)
        if selected_model == 'nb' or selected_model == 'all':
            result['nb'] = predict_sentiment(input_text, nb_model)
    
    return render_template('index.html', 
                           input_text=input_text,
                           result=result,
                           selected_model=selected_model,
                           model_accuracies=model_accuracies)   

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7877)