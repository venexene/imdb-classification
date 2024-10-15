from django.urls import path

from . import views

urlpatterns = [
    path("", views.classification_review, name="index"),
]
