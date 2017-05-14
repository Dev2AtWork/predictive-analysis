
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from restapp.serializers import housePropertySerializer
from restapp.models import houseProperty
from restapp.mb_data_analytics_func import train
from restapp.mb_data_analytics_func import Predict
from restapp.mb_data_analytics_func import reTrain
from restapp.mb_data_analytics_func import getProblemStatements
import json
from django.http import JsonResponse

class MachineLEarningPredict(APIView):

 def get(self, request, format=None):
      return Response(1+1)

 def post(self, request, format=None):
        predict=Predict(request.data["_QueryString"])
        return Response(predict)
        

class MachineLearningTrain(APIView):

 def get(self, request, format=None):
      return Response(1+1)

 def post(self, request, format=None):       
        ModelTrainData=train(request.data["_filepath"],request.data["_problemstatement"])
        return JsonResponse(ModelTrainData, safe=False)


class MachineLearningRetrain(APIView):

 def get(self, request, format=None):
      return Response(1+1)

 def post(self, request, format=None):       
        ModelTrainData=train(request.data["_filepath"],request.data["_problemstatement"])
        return JsonResponse(ModelTrainData, safe=False)

class MachineLearningProblems(APIView):

 def get(self, request, format=None):
       TrainedModelData=getProblemStatements()
       return JsonResponse(TrainedModelData,safe=False)

 def post(self, request, format=None):
        return Response(1+1)



# response={}
# response['options_list']= serializers.serialize('json',TrainedModelData)
# return HttpResponse(response,content_type="application/json")
#hpObject = houseProperty.objects.all()
      #serializer = housePropertySerializer(hpObject)
#return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
 #hpObject = houseProperty.objects.all()
       #serializer = housePropertySerializer(hpObject)
        #price=PredictPrice(request.data["_bedroom"] ,request.data["_bathroom"])
        #if serializer.is_valid():
           #serializer.save()