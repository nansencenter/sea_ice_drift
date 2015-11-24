from setuptools import setup

import_error_msg = "Sea Ice Drift requires %s, which should be installed separately"

setup(
    name = "sea_ice_drift",
    version = "0.1",
    author = ('Stefan Muckenhuber', 'Anton Korosov'),
    author_email = "stefan.muckenhuber@nersc.no",
    description = ("Drift of sea ice from satellite data using feature tracking methods"),
    long_description = ("Drift of sea ice from satellite data using feature tracking methods"),
    license = "GNU General Public License v3",
    keywords = "sar, feature tracking, ice drift",
    url = "https://github.com/nansencenter/sea_ice_drift",
    packages=['sea_ice_drift'],
    package_data={'sea_ice_drift': ['sea_ice_drift/testdata/*']},
    test_suite="sea_ice_drift.tests",
)
