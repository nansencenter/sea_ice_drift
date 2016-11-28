# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 09:57:09 2015

@author: stemuc
"""

# load ft_mcc_functions
execfile('./FT_MCC/ft_mcc_08_functions.py')

idir = '...' # Location of satellite imagery

# Image pair Fram Strait
file1 = os.path.join(idir, 'S1A_EW_GRDM_1SDH_20150328T074433_20150328T074533_005229_0069A8_801E.zip')
file2 = os.path.join(idir, 'S1A_EW_GRDM_1SDH_20150329T163452_20150329T163552_005249_006A15_FD89.zip')

# Load satellite images as Nansat functions and get initial rotation alpha
n1, n2, img1, img2, alpha = get_n_img(file1, file2, band='HV', resize_factor=0.5, maxSpeed=0.5)

# Perform feature-tracking
x1raw, y1raw, x2raw, y2raw = get_icedrift_ft(n1, n2, img1, img2, resize_factor=0.5, maxSpeed=0.5)


# perform MCC on grid
lon1_fg, lat1_fg, lon2_fg, lat2_fg, lon1_mcc, lat1_mcc, lon2_mcc, lat2_mcc, mccr = get_icedrift_mcc_grid(n1, n2, img1, img2, alpha, x1raw, y1raw, x2raw, y2raw,
                                                                                                   hws_n1 = 35, hws_n2_range = [20, 125], minMCC = 0.35,
                                                                                                   grid_step = 100, angles = np.arange(-10, 10.1, 2))


# Plot first guess and border matrix
plt = plot_FG_border(x1raw, y1raw, x2raw, y2raw, img1, hws_n1 = 35, hws_n2_range = [20, 125])
plt.show()


###############################################################################
# Plot map incl. drift vectors

import matplotlib
import matplotlib.pyplot as plt
import glob
from matplotlib import pylab
import scipy.io
from mpl_toolkits.basemap import Basemap#, cm
from matplotlib.patches import Polygon
from matplotlib.pylab import *
import matplotlib.cm as cm

def draw_screen_poly( lons, lats, m, color='black' , color_edge='black'):
    x, y = m( lons, lats )
    xy = zip(x,y)
    poly = Polygon( xy, facecolor=color,edgecolor=color_edge, alpha=0.4 )
    plt.gca().add_patch(poly)
    
def vec_from_geo(geometry):
    # Get Geometry inside Geometry
    ring=geometry.GetGeometryRef(0)
    # Write points in Vectors
    lons = []; lats = []#; pointsZ = []
    for p in range(ring.GetPointCount()):
            lon, lat, z = ring.GetPoint(p)
            lons.append(lon)
            lats.append(lat)
    #        pointsZ.append(z) 
    return lons, lats #,z

# Geometry of SAR images
sar_geometry_01=n1.get_border_geometry()
sar_geometry_02=n2.get_border_geometry()
lons_sar_01,lats_sar_01=vec_from_geo(sar_geometry_01)
lons_sar_02,lats_sar_02=vec_from_geo(sar_geometry_02)
# overlapping Polygon
IntersectBorderGeometry=n1.get_border_geometry().Intersection(n2.get_border_geometry())
lons_over,lats_over=vec_from_geo(IntersectBorderGeometry)

# figure
plt.figure(num=1, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
#m = Basemap(projection='npstere',boundinglat=69,lon_0=0,resolution='l')
lon_01=-15; lon_02=23
lat_01=77.5; lat_02=83
m = Basemap(projection='stere',lon_0=(lon_02+lon_01)/2,lat_0=(lat_02+lat_01)/2,llcrnrlon=lon_01,llcrnrlat=lat_01,urcrnrlon=lon_02,urcrnrlat=lat_02,resolution='h')
m.drawcoastlines()
m.fillcontinents(color='gray',lake_color='white')
# draw parallels and meridians.
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(-180.,181.,20.))
m.drawmapboundary(fill_color='white')
draw_screen_poly( lons_sar_01, lats_sar_01, m, 'gray' )
draw_screen_poly( lons_sar_02, lats_sar_02, m, 'gray' )
#draw_screen_poly( lons_over, lats_over, m, 'red' )


# DRIFT
# first guess
for i in range(len(lon1_fg)):
    x_01,y_01=m(lon1_fg[i],lat1_fg[i])
    x_02,y_02=m(lon2_fg[i],lat2_fg[i])
    plt.gca().arrow(x_01,y_01,x_02-x_01,y_02-y_01,width=50,lw=1,color='black')

# MCC
min_val = minMCC
max_val = 1
my_cmap = cm.get_cmap('YlOrRd')
norm = matplotlib.colors.Normalize(min_val, max_val)
for i in range(len(lon1_mcc)):
    x_01,y_01 = m(lon1_mcc[i],lat1_mcc[i])
    x_02,y_02 = m(lon2_mcc[i],lat2_mcc[i])
    color_i = my_cmap(norm(mccr[i]))
    plt.gca().arrow(x_01,y_01,x_02-x_01,y_02-y_01,width=50,lw=1,color=color_i)
        
cmmapable = cm.ScalarMappable(norm, my_cmap)
cmmapable.set_array([0])
ax = gca()
cbar = colorbar(cmmapable)
cbar.set_label('MCC')
#cbar.ax.set_yticklabels(['0','0.2','0.4','0.6','0.8','1'])

# Velocity scale
lon1_scale=-14
lat1_scale=77.9
lon2_scale=-12.765
lat2_scale=77.979
x_01,y_01=m(lon1_scale,lat1_scale)
x_02,y_02=m(lon2_scale,lat2_scale)
plt.gca().arrow(x_01,y_01,x_02-x_01,y_02-y_01,width=50,lw=3,color='black')
plt.text(x_01-3000,y_01+8000,'30km',fontsize=12,fontweight='bold',color='black')


#plt.title(str(len(lon1_mcc))+' vectors using FT + MCC')
plt.show()




