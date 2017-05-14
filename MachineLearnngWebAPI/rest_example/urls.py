from django.conf.urls import  include, url

from restapp import views

from django.contrib import admin
admin.autodiscover()

urlpatterns = {
    # Examples:
    # url(r'^$', 'rest_example.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),
    url(r'^train/', views.MachineLearningTrain.as_view()),
    url(r'^predict/', views.MachineLEarningPredict.as_view()),
    url(r'^problems/', views.MachineLearningProblems.as_view()),
    url(r'^retrain/', views.MachineLearningRetrain.as_view()),    
}
