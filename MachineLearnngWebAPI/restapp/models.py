from django.db import models

# Create your models here.
class houseProperty(models.Model):
    """This class represents the House Property model."""
    _suburb = models.CharField(max_length=255, blank=True, unique=False)
    #_selling_Date = models.DateTimeField(auto_now_add=False)
    #_rooms=models.IntegerField()
   # _type=models.CharField(max_length=255, blank=False, unique=True)
    #_method=models.CharField(max_length=50, blank=False, unique=True)
   # _distance=models.FloatField()
   # _postcode=models.IntegerField()
    #_bedroom=models.IntegerField()
   # _bathroom=models.IntegerField()
    #_carParking=models.IntegerField()
    #_landsize=models.IntegerField()
    #_buildingArea=models.IntegerField()
    #_yearBuilt=models.DateTimeField(auto_now_add=False)
    #_councilArea=models.CharField(max_length=255, blank=True, unique=False)
    def __str__(self):
        """Return a human readable representation of the model instance."""
        return "{}".format(self.name)