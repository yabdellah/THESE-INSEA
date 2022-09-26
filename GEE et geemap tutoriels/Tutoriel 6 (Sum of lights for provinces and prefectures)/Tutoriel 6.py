#!/usr/bin/env python
# coding: utf-8

# # Extraction de "Sum Of Lights" à partir des images NTL (Provinces et Préfectures par mois)

# In[1]:


try:
    import geemap, ee
except ModuleNotFoundError:
    if 'google.colab' in str(get_ipython()):
        print("package not found, installing w/ pip in Google Colab...")
        get_ipython().system('pip install geemap')
    else:
        print("package not found, installing w/ conda...")
        get_ipython().system('conda install mamba -c conda-forge -y')
        get_ipython().system('mamba install geemap -c conda-forge -y')
    import geemap, ee


# In[2]:


try:
    import geopandas
except ModuleNotFoundError:
    if 'google.colab' in str(get_ipython()):
        print("package not found, installing w/ pip in Google Colab...")
        get_ipython().system('pip install geopandas')
    else:
        print("package not found, installing w/ conda...")
        get_ipython().system('pip install geopandas')
    import geopandas


# In[3]:


try:
        ee.Initialize()
except Exception as e:
        ee.Authenticate()
        ee.Initialize()


# In[4]:


import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point,Polygon
import numpy as np
from functools import reduce
import shapefile
import geopandas as gpd


# In[5]:


#shapefile to FeatureCollection
def upload_shapefile_to_gee1(shp_file):
    """
    Upload a shapefile to Google Earth Engine as an asset.

    Args:
        user (django.contrib.auth.User): the request user.
        shp_file (shapefile.Reader): A shapefile reader object.
    """
    features = []
    fields = shp_file.fields[1:]
    field_names = [field[0] for field in fields]

    # Convert Shapefile to ee.Features
    for record in shp_file.shapeRecords():
        # First convert to geojson
        attributes = dict(zip(field_names, record.record))
        geojson_geom = record.shape.__geo_interface__
        geojson_feature = {
            'type': 'Feature',
            'geometry': geojson_geom,
            'properties': attributes
        }

        # Create ee.Feature from geojson (this is the Upload, b/c ee.Feature is a server object)
        features.append(ee.Feature(geojson_feature))

    feature_collection = ee.FeatureCollection(features)
    return feature_collection


# In[6]:


#print a shape file (geojson)
def upload_shapefile_to_gee(shp_file):
    """
    Upload a shapefile to Google Earth Engine as an asset.

    Args:
        user (django.contrib.auth.User): the request user.
        shp_file (shapefile.Reader): A shapefile reader object.
    """
    features = []
    fields = shp_file.fields[1:]
    field_names = [field[0] for field in fields]

    # Convert Shapefile to ee.Features
    for record in shp_file.shapeRecords():
        # First convert to geojson
        attributes = dict(zip(field_names, record.record))
        geojson_geom = record.shape.__geo_interface__
        geojson_feature = {
            'type': 'Feature',
            'geometry': geojson_geom,
            'properties': attributes
        }

        print(geojson_feature)


# In[7]:


import shapefile


# In[9]:


#sf = shapefile.Reader("/content/Provinces & Préfecture.shp")
#shapeRecs = sf.shapeRecords()
#shapeRecs.__geo_interface__['type']
#upload_shapefile_to_gee(sf)
#shapes = sf.shapes()
#shapes


# In[10]:


#shapefile = gpd.read_file("/content/Provinces & Préfecture.shp")
#print(shapefile)
#shapefile.head(12)
#print(shapeRecs.__geo_interface__)


# In[11]:


# revise our reducer function to be to get SOL for morocco
def get_morocco_sol(img):
    sol = img.reduceRegion(reducer=ee.Reducer.sum(), geometry=morocco00, scale=500, maxPixels=1e20).get('avg_rad') #'avg_rad' for viirs
    return img.set('date', img.date().format()).set('SOL',sol)


# In[12]:


def Sol(s,Date1,Date2,Satelite_Data):
    global morocco00
    
    viirs = ee.ImageCollection(Satelite_Data).filterDate(Date1,Date2)
    morocco00 = ee.FeatureCollection(upload_shapefile_to_gee1(sf)).filter(ee.Filter.eq('Name', s)).first().geometry()
    get_morocco_sol
    morocco00_sol = viirs.map(get_morocco_sol)

    # get lists
    nested_list = morocco00_sol.reduceColumns(ee.Reducer.toList(2), ['date','SOL']).values().get(0)

    # convert to dataframe
    soldf = pd.DataFrame(nested_list.getInfo(), columns=['date','SOL'])
    soldf = soldf.rename(columns={'SOL': s})
    return soldf


# In[16]:


Date1='2014-01-01'
Date2='2020-12-01'


# In[ ]:


Provinces = ['Al Haouz', 'Al Hoceima', 'Aousserd', 'Assa Zag', 'Azilal', 'benimellal', 'Benslimane', 'Berkane', 'Berrechid', 'Boujdour', 'Boulemane', 'Casablanca', 'Chefchaouen', 'Chichaoua', 'Chtouka Ait Baha' , 'Driouch', 'El Hajeb', 'El Jadida', 'El Kelaa des Sraghna', 'Errachidia', 'Es Semara', 'Essaouira', 'Fahs Anjra', 'Fes', 'Figuig', 'Fquih Ben Salah', 'Guelmim', 'Guercif', 'Ifrane', 'Inezgane Ait Melloul', 'Jerada', 'kenitra', 'Khemisset', 'khenifra', 'Khouribga', 'laayoune', 'Larache', "M'diq", 'Marrakech', 'Mediouna', 'Meknes', 'midelt', 'Mohammedia', 'Moulay Yacoub', 'Nador', 'Nouaceur', 'Ouarzazate', 'Oued Eddahab', 'Ouezan', 'Oujda Angad', 'Rabat', 'Rehamna', 'Safi', 'Sale', 'Sefrou', 'Settat', 'sidi benour', 'Sidi Ifni', 'Sidi Kacem', 'Sidi Slimane', 'Skhirate Temara', 'Tan Tan', 'Tanger Assilah', 'Taounate', 'Taourirt', 'Tarfaya', 'Taroudannt', 'Tata', 'Taza', 'Tetouan', 'tinghir', 'Tiznit', 'Youssoufia', 'Zagora']
#Regions = ['Tanger-Tétouan-Al Hoceima', 'Fès-Meknès', 'Beni Mellal-Khénifra', 'Rabat-Salé-Kénitra', 'Casablanca-Settat', 'Marrakech-Safi', 'Draa-Tafilalet', 'Souss-Massa', 'Guelmim-Oued Noun', 'Laayoune-Sakia-El-Hamra', 'Dakhla-Oued Ed-Dahab']


# In[17]:


Provinces_et_Préfectures = ["Province d'Ifrane ⵜⴰⵙⴳⴰ ⵏ ⵉⴼⵔⴰⵏ إقليم إفران","Province d'Es-Semara إقليم السمارة","Province d'Errachidia إقليم الرشيدية","Province de Berkane إقليم بركان","Province de Benslimane إقليم بن سليمان","Province de Beni Mellal إقليم بني ملال","Province d'Ouezzane إقليم وزان","Province d'Aousserd إقليم أوسرد","Province d'Al Hoceima إقليم الحسيمة","Province d'Al Haouz ⵍⵉⵇⵍⵉⵎ ⵏ ⵍⵃⵓⵣ إقليم الحوز","Province Assa-Zag ⵍⵉⵇⵍⵉⵎ ⵏ ⴰⵙⵙⴰ ⵣⴰⴳ إقليم آسا الزاك","Province d'El Kelâat Es-Sraghna إقليم قلعة السراغنة","Province d'El Jadida إقليم الجديدة","Province d'El Hajeb إقليم الحاجب","Province d'Azilal إقليم أزيلال","Province de Guercif إقليم جرسيف","Province de Guelmim ⵍⵉⵇⵍⵉⵎ ⵏ ⴳⵍⵎⵉⵎ إقليم كلميم","Province de Fquih Ben Saleh إقليم الفقيه بن صالح","Province de Figuig إقليم الناظور","Province de Khénifra إقليم خنيفرة","Province de Khémisset إقليم الخميسات","Province de Kenitra إقليم القنيطرة","Province de Jerada إقليم جرادة","Province de Chefchaouen إقليم شفشاون","Province de Boulemane إقليم بولمان","Province de Boujdour إقليم بوجدور","Province de Berrechid إقليم برشيد","Province de Fahs-Anjra إقليم الفحص-أنجرة","Province de Driouch إقليم الدريوش","Province de Chtouka Aït Baha ⵍⵉⵇⵍⵉⵎ ⵏ ⵛⵜⵓⴽⴰ ⴰⵢⵜ ⴱⴰⵀⴰ إقليم شتوكة آيت باها","Province de Chichaoua ⵍⵉⵇⵍⵉⵎ ⵏ ⵛⵉⵛⴰⵡⴰ إقليم شيشاوة","Province de Settat إقليم سطات","Province de Sefrou إقليم صفرو","Province de Safi إقليم أسفي","Province de Rhamna إقليم الرحامنة","Province de Sidi Slimane إقليم سيدي سليمان","Province de Sidi Kacem إقليم سيدي قاسم","Province de Sidi Ifni ⵍⵉⵇⵍⵉⵎ ⵏ ⵙⵉⴷⵉ ⵉⴼⵏⵉ إقليم سيدي إفني","Province de Sidi Bennour إقليم سيدي بنور","Province de Médiouna إقليم مديونة","Province de Larache إقليم العرائش","Province de Laâyoune إقليم العيون","Province de Khouribga إقليم خريبكة","Province de Ouarzazate ⵍⵉⵇⵍⵉⵎ ⵏ ⵡⴰⵔⵣⴰⵣⴰⵜ إقليم ورززات","Province de Nador إقليم الناظور","Province de Moulay Yacoub إقليم مولاي يعقوب","Province de Midelt إقليم ميدلت","Province de Zagora ⵍⵉⵇⵍⵉⵎ ⵏ ⵣⴰⴳⵓⵔⴰ إقليم زاكورة","Province de Youssoufia إقليم اليوسفية","Province de Tiznit ⵍⵉⵇⵍⵉⵎ ⵏ ⵜⵉⵣⵏⵉⵜ إقليم تزنيت","Province de Tinghir إقليم تنغير","Préfecture d'Inezgane-Aït Melloul ⵍⵄⴰⵎⴰⵍⴰ ⵏ ⵉⵏⵣⴳⴰⵏ-ⴰⵢⵜ ⴺⵍⵓⵍ عمالة إنزكان آيت ملول","Préfecture d'Agadir Ida-Outanane ⵍⵄⴰⵎⴰⵍⴰ ⵏ ⴰⴳⴰⴷⵉⵔ ⵉⴷⴰ ⵡⵜⴰⵏⴰⵏ عمالة أكادير إدا وتنان","Province Nouaceur إقليم النواصر","Province Essaouira ⵍⵉⵇⵍⵉⵎ ⵏ ⵚⵡⵉⵔⴰ إقليم الصويرة‎","Province de Tarfaya إقليم طرفاية","Province de Taourirt إقليم تاوريرت","Province de Taounate إقليم تاونات","Province de Tan-Tan إقليم طانطان","Province de Tétouan إقليم تطوان","Province de Taza اقليم تازة","Province de Tata ⵍⵉⵇⵍⵉⵎ ⵏ ⵟⴰⵟⴰ إقليم طاطا","Province de Taroudant ⵍⵉⵇⵍⵉⵎ ⵏ ⵜⴰⵔⵓⴷⴰⵏⵜ إقليم تارودانت","Préfecture de Tanger-Assilah عمالة طنجة-أصيلة","Préfecture de Skhirate-Témara عمالة الصخيرات-تمارة","Préfecture de Salé عمالة سلا","Préfecture de M'diq-Fnideq عمالة المضيق الفنيدق","Préfecture de Fès عمالة فاس","Préfecture de Casablanca عمالة الدار البيضاء","Préfecture d'Oujda-Angad عمالة وجدة - أنجاد","Préfecture de Rabat عمالة الرباط","Préfecture de Mohammédia عمالة المحمدية","Prefecture de Meknès عمالة مكناس","Préfecture de Marrakech عمالة مراكش"]


# In[18]:


len(Provinces_et_Préfectures)


# In[19]:


VIIRS = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG" #2014-01-01T00:00:00 - 2020-12-01T00:00:00
DMSP = "NOAA/DMSP-OLS/CALIBRATED_LIGHTS_V4" #1996-03-16T00:00:00 - 2011-07-31T00:00:00


# In[20]:


#D1=Sol('Oriental', Date1, Date2, VIIRS)
D1=Sol("Province d'Oued Ed-Dahab إقليم وادي الذهب", Date1, Date2, VIIRS)  #'Oriental' for regions
D1


# In[21]:


D1


# In[22]:


def qwe(list):
    D1=Sol("Province d'Oued Ed-Dahab إقليم وادي الذهب",Date1,Date2,VIIRS)
    D=D1
    for i in list :
        D=D.join(Sol(i,Date1,Date2,VIIRS)[i], on=None, how='left', lsuffix='', rsuffix='', sort=False)
    return D


# In[23]:


Data=qwe(Provinces_et_Préfectures)
Data


# In[24]:


Data['date'] = pd.to_datetime(Data['date'])


# In[25]:


Data.to_csv(r'/content/MyData.csv')

