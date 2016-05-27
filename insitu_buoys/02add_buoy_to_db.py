import os, sys, glob
import datetime

import numpy as np
import matplotlib.pyplot as plt

import pythesint as pti

os.environ['DJANGO_SETTINGS_MODULE'] = 'project.settings'
sys.path.insert(0, 'project')

import django
django.setup()

from django.contrib.gis.geos import LineString

from nansencloud.catalog.models import Dataset
from nansencloud.vocabularies.models import Platform
from nansencloud.vocabularies.models import Instrument
from nansencloud.vocabularies.models import DataCenter
from nansencloud.vocabularies.models import ISOTopicCategory
from nansencloud.catalog.models import GeographicLocation
from nansencloud.catalog.models import DatasetURI, Source, Dataset

fnames = glob.glob('winter/*.csv')

def get_dates_lon_lat(fname):
    ''' Get valid dates, lon and lat arrays from file'''
    data = np.recfromcsv(fname, invalid_raise=False, delimiter=',',names=True)
    gpi = data['lat'] > 0
    valid_dates = data['date'][gpi]
    valid_lon = data['lon'][gpi]
    valid_lat = data['lat'][gpi]

    return valid_dates, valid_lon, valid_lat

fname = fnames[0]
dates, lon, lat = get_dates_lon_lat(fname)
line1 = LineString( (zip(lon, lat)) )


