# -*- coding: utf-8 -*-
"""
Created on 17 Nov 2015
@author: Stefan Muckenhuber et al. (NERSC)
'Sea ice drift from Sentinel-1 SAR imagery using open source feature tracking'
"""

# Import sea ice drift function
execfile('s1a_icedrift.py')

# Define Sentinel-1 image pair
file_a = 'S1A_EW_GRDM_1SDH_20150328T074433_20150328T074533_005229_0069A8_801E.SAFE'
file_b = 'S1A_EW_GRDM_1SDH_20150329T163452_20150329T163552_005249_006A15_FD89.SAFE'

# Load Sentinel-1 images as Nansat objects
n_a=Nansat(file_a)
n_b=Nansat(file_b)

# Apply sea ice drift function on image pair using either a) HV or b) HV+HH polarisation:

# a) HV
query_lonlat, train_lonlat=s1a_icedrift(n_a,n_b)

## b) HH+HV
#query_lonlat_HH, train_lonlat_HH = s1a_icedrift(n_a,n_b,band='HH')#,nFeatures=100000,patchSize=34,ratio_test=0.8,resize_factor=0.5)
#query_lonlat_HV, train_lonlat_HV = s1a_icedrift(n_a,n_b,band='HV')
#query_lonlat = np.column_stack((query_lonlat_HH,query_lonlat_HV))
#train_lonlat = np.column_stack((train_lonlat_HH,train_lonlat_HV))



############################################################################### Plot results

import matplotlib.pyplot as plt
import glob
from matplotlib import pylab
import scipy.io
from mpl_toolkits.basemap import Basemap, cm
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


x = query_lonlat[0]
y = query_lonlat[1]

# speed in meters
u = []
v = []
for i in range(query_lonlat[0].size):
    # Equirectangular approximation
    dlong = (train_lonlat[0][i] - query_lonlat[0][i])*np.pi/180;
    dlat  = (train_lonlat[1][i] - query_lonlat[1][i])*np.pi/180;
    slat  = (train_lonlat[1][i] + query_lonlat[1][i])*np.pi/180;
    p1 = (dlong)*np.cos(0.5*slat)
    p2 = (dlat)
    u.append(6371000 * p1)
    v.append(6371000 * p2)

u=np.array(u)
v=np.array(v)

# introduce grid
x_lim=[-13,8.1]
y_lim=[79,83.1]
step_lon = 1
step_lat = 0.2
x_grid = np.arange(x_lim[0], x_lim[1], step_lon)
y_grid = np.arange(y_lim[0], y_lim[1], step_lat)

# grid data
u_grid = []
v_grid = []
u_std_grid = []
v_std_grid = []
n_grid = []
rmsd_grid = []

for i in range(len(y_grid)-1):
    u_row = []
    v_row = []
    u_std_row = []
    v_std_row = []
    n_row = []
    rmsd_row = []
    for j in range(len(x_grid)-1):
        # filter inside grid box, query needs to be inside box
        inside_grid = (x >= x_grid[j]) * (x <= x_grid[j+1]) * (y >= y_grid[i]) * (y <= y_grid[i+1])

        u_row.append(np.mean(u[inside_grid]))
        v_row.append(np.mean(v[inside_grid]))
        u_std_row.append(np.std(u[inside_grid]))
        v_std_row.append(np.std(v[inside_grid]))
        n_row.append(len(u[inside_grid]))

        u_m=np.mean(u[inside_grid])
        v_m=np.mean(v[inside_grid])
        u_i=u[inside_grid]
        v_i=v[inside_grid]
        rd_i=[]
        for k in range(len(u_i)):
            rd_i.append((u_m-u_i[k])**2+(v_m-v_i[k])**2)
        if len(rd_i) > 0:
            rmsd_point=np.sqrt(sum(rd_i)/len(rd_i))
        else:
            rmsd_point=0
        rmsd_row.append(rmsd_point)

    u_grid.append(np.array(u_row))
    v_grid.append(np.array(v_row))
    u_std_grid.append(np.array(u_std_row))
    v_std_grid.append(np.array(v_std_row))
    n_grid.append(np.array(n_row))
    rmsd_grid.append(np.array(rmsd_row))


# Geometry of SAR images
sar_geometry_01=n_a.get_border_geometry()
sar_geometry_02=n_b.get_border_geometry()
lons_sar_01,lats_sar_01=vec_from_geo(sar_geometry_01)
lons_sar_02,lats_sar_02=vec_from_geo(sar_geometry_02)
# overlapping Polygon
IntersectBorderGeometry=n_a.get_border_geometry().Intersection(n_b.get_border_geometry())
lons_over,lats_over=vec_from_geo(IntersectBorderGeometry)


############################################################################### Plot DRIFT
# figure
plt.figure(num=1, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
#m = Basemap(projection='npstere',boundinglat=69,lon_0=0,resolution='l')
lon_01=-15; lon_02=23
lat_01=77.5; lat_02=83
m = Basemap(projection='stere',lon_0=(lon_02+lon_01)/2,lat_0=(lat_02+lat_01)/2,llcrnrlon=lon_01,llcrnrlat=lat_01,urcrnrlon=lon_02,urcrnrlat=lat_02,resolution='h')
m.drawcoastlines()
m.fillcontinents(color='lightgreen',lake_color='lightblue')
# draw parallels and meridians.
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(-180.,181.,20.))
m.drawmapboundary(fill_color='lightblue')

draw_screen_poly( lons_sar_01, lats_sar_01, m, 'black' )
draw_screen_poly( lons_sar_02, lats_sar_02, m, 'black' )
# plot overlap
draw_screen_poly( lons_over, lats_over, m, 'red' )
# DRIFT
for i in range(query_lonlat[0].size):
    # start
    x_01,y_01=m(query_lonlat[0][i],query_lonlat[1][i])
    # end
    x_02,y_02=m(train_lonlat[0][i],train_lonlat[1][i])
    plt.gca().arrow(x_01,y_01,x_02-x_01,y_02-y_01,width=50,lw=2,color='red')
plt.title(str(query_lonlat[0].size)+" vectors")
plt.show()


############################################################################### Plot N
# figure
plt.figure(num=3, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
#m = Basemap(projection='npstere',boundinglat=69,lon_0=0,resolution='l')
lon_01=-15; lon_02=23
lat_01=77.5; lat_02=83
m = Basemap(projection='stere',lon_0=(lon_02+lon_01)/2,lat_0=(lat_02+lat_01)/2,llcrnrlon=lon_01,llcrnrlat=lat_01,urcrnrlon=lon_02,urcrnrlat=lat_02,resolution='h')
m.drawcoastlines()
m.fillcontinents(color='lightgreen',lake_color='lightblue')
# draw parallels and meridians.
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(-180.,181.,20.))
m.drawmapboundary(fill_color='lightblue')

min_val = 0#np.min(n_grid)
max_val = 50#np.max(n_grid)
my_cmap = cm.get_cmap('YlGn') # or any other one
norm = matplotlib.colors.Normalize(min_val, max_val) # the color maps work for [0, 1]
# number of vectors
for i in range(len(x_grid)-1):
    for j in range(len(y_grid)-1):
        lons = [ x_grid[i], x_grid[i], x_grid[i+1], x_grid[i+1] ]
        lats = [ y_grid[j], y_grid[j+1], y_grid[j+1], y_grid[j] ]
        x_i = n_grid[j][i]
        color_i = my_cmap(norm(x_i)) # returns an rgba value
        draw_screen_poly( lons, lats, m , color=color_i, color_edge='gray')
        if x_i==0:
            draw_screen_poly( lons, lats, m , color='gray', color_edge='gray')

cmmapable = cm.ScalarMappable(norm, my_cmap)
cmmapable.set_array(range(min_val, max_val))
ax = gca()
cbar = colorbar(cmmapable)
cbar.set_label('Number of vectors')
cbar.ax.set_yticklabels(['0','5','10','15','20','25','30','35','40','45','> 50'])
plt.show()


############################################################################### plot RMSD
# figure
plt.figure(num=4, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
#m = Basemap(projection='npstere',boundinglat=69,lon_0=0,resolution='l')
lon_01=-15; lon_02=23
lat_01=77.5; lat_02=83
m = Basemap(projection='stere',lon_0=(lon_02+lon_01)/2,lat_0=(lat_02+lat_01)/2,llcrnrlon=lon_01,llcrnrlat=lat_01,urcrnrlon=lon_02,urcrnrlat=lat_02,resolution='h')
m.drawcoastlines()
m.fillcontinents(color='lightgreen',lake_color='lightblue')
# draw parallels and meridians.
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(-180.,181.,20.))
m.drawmapboundary(fill_color='lightblue')

min_val = 0#np.min(rmsd_grid)
max_val = 3#np.max(rmsd_grid)
my_cmap = cm.get_cmap('Reds') # or any other one
norm = matplotlib.colors.Normalize(min_val, max_val) # the color maps work for [0, 1]
# number of vectors
for i in range(len(x_grid)-1):
    for j in range(len(y_grid)-1):
        lons = [ x_grid[i], x_grid[i], x_grid[i+1], x_grid[i+1] ]
        lats = [ y_grid[j], y_grid[j+1], y_grid[j+1], y_grid[j] ]
        x_i = rmsd_grid[j][i]/1000.
        color_i = my_cmap(norm(x_i)) # returns an rgba value
        draw_screen_poly( lons, lats, m , color=color_i, color_edge='gray')
        if x_i==0:
            draw_screen_poly( lons, lats, m , color='gray', color_edge='gray')

cmmapable = cm.ScalarMappable(norm, my_cmap)
cmmapable.set_array(range(min_val, max_val))
ax = gca()
cbar = colorbar(cmmapable)
cbar.set_label('RMSD [km]')
cbar.ax.set_yticklabels(['0','0.3','0.6','0.9','1.2','1.5','1.8','2.1','2.4','2.7','> 3'])
plt.show()



