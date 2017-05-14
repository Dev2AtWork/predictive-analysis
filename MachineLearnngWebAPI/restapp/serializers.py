from django.contrib.auth.models import User
from .models import houseProperty
from rest_framework import serializers


class housePropertySerializer(serializers.ModelSerializer):
    class Meta:
        model = houseProperty
        fields = ('_suburb',)
 #       fields = ('_suburb','_rooms', '_type','_method','_distance','_postcode','_bedroom','_bathroom','_carParking','_landsize','_buildingArea','_councilArea')
