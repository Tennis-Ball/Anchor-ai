from django.urls import path
from . import views


urlpatterns = [
    path('home/<str:symbol>/', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('markets/<str:symbol>/', views.markets, name='markets'),
    path('', views.landing_redirect, name='redirect')
]
