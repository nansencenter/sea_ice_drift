from setuptools import setup

import_error_msg = "Sea Ice Drift requires %s, which should be installed separately"

setup(
    name = "sea_ice_drift",
    version = "0.2",
    author = ('Anton Korosov', 'Stefan Muckenhuber'),
    author_email = "anton.korosov@nersc.no",
    description = ("Drift of sea ice from satellite data using feature tracking and pattern matching methods"),
    long_description = ("Drift of sea ice from satellite data using feature tracking and pattern matching methods"),
    license = "GNU General Public License v3",
    keywords = "sar, feature tracking, pattern matching, ice drift",
    url = "https://github.com/nansencenter/sea_ice_drift",
    packages=['sea_ice_drift'],
    test_suite="sea_ice_drift.tests",
)
