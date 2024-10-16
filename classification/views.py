from django.shortcuts import render
from django.http import JsonResponse
from nltk.stem.snowball import SnowballStemmer
import pickle
import re

f = open('stopwords.txt', 'r')
stopwords_list = f.readlines()
for i in range(len(stopwords_list)):
    stopwords_list[i] = stopwords_list[i].replace('\n', '')
stopwords = set(stopwords_list)

vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", 'rb'))
model = pickle.load(open("models/sgd_model.pkl", 'rb'))

def preprocess(txt):
    txt = re.sub('<[^>]*>', '', txt)
    emots = re.findall(r'(?::|;|=) (?:-)?(?:\)|\(|D|P)', txt)
    txt = (re.sub(r'[\W]+', ' ', txt.lower()) + ' '.join(emots).replace('-', ''))
    stemmer = SnowballStemmer("english")
    txt = ' '.join([stemmer.stem(word) for word in txt.split()])
    txt = ' '.join([word for word in txt.split() if word not in stopwords])
    return txt

def classification_review(request):
    sentiment = None
    rating = None
    if request.method == 'POST':
        review_text = request.POST['review_text']
        review_text = preprocess(review_text)
        review_vector = vectorizer.transform([review_text])
        prediction = model.predict(review_vector)[0]
        prob = model.predict_proba(review_vector)[0, 1]
        sentiment = 'Положительный' if prediction == 1 else 'Отрицательный'
        rating = round(prob * 9 + 1, 2)
        return JsonResponse({'sentiment': sentiment, 'rating': rating})
    return render(request, 'predict.html', {'sentiment': sentiment, 'rating': rating})
