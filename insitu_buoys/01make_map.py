import glob
import numpy as np
import matplotlib.pyplot as plt

date0 = np.datetime64('2015-01-01 00:00:00')
date_step = np.timedelta64(10*24*60*60, 's')


while date0 < np.datetime64('2015-05-30 00:00:00'):
    date1 = date0 + date_step

    idir = 'winter'
    fnames = glob.glob(idir + '/*.csv')
    lons = []
    lats = []
    for fname in fnames:
        print fname
        data = np.recfromcsv(fname, invalid_raise=False, delimiter=',',names=True)
        dates = np.array([np.datetime64(date) for date in data['date']])
        gpi = (data['lat'] > 0) * (dates >= date0) * (dates < date1)
        #plt.plot(data['lon'][gpi], data['lat'][gpi], '.-')
        #plt.plot(dates[gpi], data['u_ms'][gpi])
        lons.append(data['lon'][gpi])
        lats.append(data['lat'][gpi])

    if np.hstack(lons).size > 0:
        print np.hstack(lons).min(), np.hstack(lons).max()
        for lon, lat in zip(lons, lats):
            plt.plot(lon, lat, '.-')
        plt.show()

    date0 = date1
