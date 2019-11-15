from django.urls import path
from . import views
from django.views.generic import RedirectView


urlpatterns = [
    path('', views.get_image, name='get_image')

]