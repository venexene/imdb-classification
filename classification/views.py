from django.shortcuts import render
from django.http import JsonResponse
import pickle

vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", 'rb'))
model = pickle.load(open("models/lr_model.pkl", 'rb'))

def classification_review(request):
    sentiment = None
    rating = None
    if request.method == 'POST':
        review_text = request.POST['review_text']
        review_vector = vectorizer.transform([review_text])
        prediction = model.predict(review_vector)[0]
        prob = model.predict_proba(review_vector)[0, 1]
        sentiment = 'Положительный' if prediction == 1 else 'Отрицательный'
        rating = round(prob * 9 + 1, 2)
        return JsonResponse({'sentiment': sentiment, 'rating': rating})
    return render(request, 'predict.html', {'sentiment': sentiment, 'rating': rating})
