import os


import matplotlib
# disable interactive to speed up
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import glob
from matplotlib import pylab
import scipy.io
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.patches import Polygon
from matplotlib.pylab import *
import matplotlib.cm as cm

from nansat import Nansat

from sea_ice_drift import *

def draw_screen_poly( lons, lats, m, color='black' , color_edge='black'):
    ''' Draw polygon '''
    x, y = m( lons, lats )
    xy = zip(x,y)
    poly = Polygon( xy, facecolor=color,edgecolor=color_edge, alpha=0.4 )
    plt.gca().add_patch(poly)

def make_basemap(lons_sar_01, lats_sar_01,
                 lons_sar_02, lats_sar_02,
                 lons_over, lats_over,
                 lon_0, lat_0,
                 width=800000, height=700000,
                 resolution='c'):
    ''' Create Basemap for plotting '''
    # figure
    plt.figure(num=1, figsize=(10, 7), dpi=300, facecolor='w', edgecolor='k')
    #m = Basemap(projection='npstere',boundinglat=69,lon_0=0,resolution='l')
    m = Basemap(projection='stere',
                lon_0=lon_0,
                lat_0=lat_0,
                width=width,
                height=height,
                resolution=resolution)
    m.drawcoastlines()
    m.fillcontinents(color='lightgreen',lake_color='lightblue')
    # draw parallels and meridians.
    m.drawparallels(np.linspace(latMin, latMax, 10),
                    labels=[1,0,0,0], fmt='%3.1f')
    m.drawmeridians(np.linspace(lonMin, lonMax, 10),
                    labels=[0,0,0,1], fmt='%3.1f')
    m.drawmapboundary(fill_color='lightblue')

    draw_screen_poly( lons_sar_01, lats_sar_01, m, 'black' )
    draw_screen_poly( lons_sar_02, lats_sar_02, m, 'black' )
    # plot overlap
    draw_screen_poly( lons_over, lats_over, m, 'red' )

    return m

# size reduction factor
factor = 0.5
# min, max of sigma0 HV (linear units)
vmin, vmax = 0, 0.013
# max allowed displacement
maxSpeed = 50

# input directory
idir = '/files/sentinel1a/safefiles/'
# Define Sentinel-1 image pair
file1 = 'S1A_EW_GRDM_1SDH_20150328T074433_20150328T074533_005229_0069A8_801E.SAFE'
file2 = 'S1A_EW_GRDM_1SDH_20150329T163452_20150329T163552_005249_006A15_FD89.SAFE'

# Load Sentinel-1 images as Nansat objects
n1 = Nansat(os.path.join(idir, file1))
n2 = Nansat(os.path.join(idir, file2))

# increase accuracy
n1 = reproject_gcp_to_stere(n1)
n2 = reproject_gcp_to_stere(n2)

# increase speed
n1.resize(factor, eResampleAlg=-1)
n2.resize(factor, eResampleAlg=-1)

# get matrices with data
img1 = n1['sigma0_HV']
img2 = n2['sigma0_HV']

# convert to 0 - 255
img1 = get_uint8_image(img1, vmin, vmax)
img2 = get_uint8_image(img2, vmin, vmax)

# find many key points
kp1, descr1 = find_key_points(img1, nFeatures=200000)
kp2, descr2 = find_key_points(img2, nFeatures=200000)

# find coordinates of matching key points
x1, y1, x2, y2 = get_match_coords(kp1, descr1, kp2, descr2)

# convert x,y to lon, lat
lon1, lat1 = n1.transform_points(x1, y1)
lon2, lat2 = n2.transform_points(x2, y2)

# find displacement in kilometers
u, v = get_displacement_km(n1, x1, y1, n2, x2, y2)

## b) HH+HV
#query_lonlat_HH, train_lonlat_HH = s1a_icedrift(n_a,n_b,band='HH')#,nFeatures=100000,patchSize=34,ratio_test=0.8,resize_factor=0.5)
#query_lonlat_HV, train_lonlat_HV = s1a_icedrift(n_a,n_b,band='HV')
#query_lonlat = np.column_stack((query_lonlat_HH,query_lonlat_HV))
#train_lonlat = np.column_stack((train_lonlat_HH,train_lonlat_HV))

# filter too high u, v
gpi = np.hypot(u, v) < maxSpeed
u = u[gpi]
v = v[gpi]
lon1 = lon1[gpi]
lat1 = lat1[gpi]
lon2 = lon2[gpi]
lat2 = lat2[gpi]


############################################################################### Plot results
# Geometry of SAR images
lons_sar_01, lats_sar_01 = n1.get_border()
lons_sar_02, lats_sar_02 = n2.get_border()
# overlapping Polygon
IntersectBorderGeometry = n1.get_border_geometry().Intersection(n2.get_border_geometry())
lons_over,lats_over = np.array(IntersectBorderGeometry.GetGeometryRef(0).GetPoints()).T

# lon, lat limits for mapping
lonMin = min(lons_sar_01.min(), lons_sar_02.min())
lonMax = max(lons_sar_01.max(), lons_sar_02.max())
latMin = min(lats_sar_01.min(), lats_sar_02.min())
latMax = max(lats_sar_01.max(), lats_sar_02.max())
lon_0 = (lonMin + lonMax) / 2
lat_0 = (latMin + latMax) / 2

# introduce grid
lonStep = 1
latStep = 0.2
lonVec = np.arange(lonMin, lonMax, lonStep)
latVec = np.arange(latMin, latMax, latStep)
lonGrid, latGrid = np.meshgrid(lonVec, latVec)

# grid data
u_grid = []
v_grid = []
u_std_grid = []
v_std_grid = []
n_grid = []
rmsd_grid = []

for lat in latVec:
    u_row = []
    v_row = []
    u_std_row = []
    v_std_row = []
    n_row = []
    rmsd_row = []
    for lon in lonVec:
        # filter inside grid box, query needs to be inside box
        inside_grid = (lon1 >= lon) * (lon1 < lon + lonStep) * (lat1 >= lat) * (lat1 < lat + latStep)

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

# convert to masked arrays
n_grid = np.array(n_grid)
n_grid = np.ma.array(n_grid, mask=(n_grid == 0))
rmsd_grid = np.array(rmsd_grid)
rmsd_grid = np.ma.array(rmsd_grid, mask=n_grid.mask)

############################################################################### 
# Plot vectors
m = make_basemap(lons_sar_01, lats_sar_01,
                 lons_sar_02, lats_sar_02,
                 lons_over,lats_over,
                 lon_0, lat_0,
                 resolution='h')
x1, y1 = m(lon1, lat1)
m.quiver(x1, y1, u, v, zorder=10, scale=1000, width=.0001)

plt.title(str(lon1.size)+" vectors")
plt.savefig('drift_vectors.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close('all')

###############################################################################
# Plot N
m = make_basemap(lons_sar_01, lats_sar_01,
                 lons_sar_02, lats_sar_02,
                 lons_over,lats_over,
                 lon_0, lat_0,
                 resolution='h')

xGrid, yGrid = m(lonGrid, latGrid)
m.pcolormesh(xGrid, yGrid, n_grid, vmin=0, vmax=250, zorder=10)
cbar = plt.colorbar()
cbar.set_label('Number of vectors')
plt.savefig('density_grid.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

###############################################################################
### plot RMSD
m = make_basemap(lons_sar_01, lats_sar_01,
                 lons_sar_02, lats_sar_02,
                 lons_over,lats_over,
                 lon_0, lat_0,
                 resolution='h')

xGrid, yGrid = m(lonGrid, latGrid)
m.pcolormesh(xGrid, yGrid, rmsd_grid, vmin=0, vmax=10, zorder=10)
cbar = plt.colorbar()
cbar.set_label('RMSD [km]')
plt.savefig('rmsd_grid.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()




