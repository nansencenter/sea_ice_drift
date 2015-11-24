from setuptools import setup

setup(
    name = "sea_ice_drift",
    version = "0.1",
    author = ('Stefan Muckenhuber', 'Anton Korosov'),
    author_email = "stefan.muckenhuber@nersc.no",
    description = ("Drift of sea ice from satellite data using feature tracking methods"),
    license = "GPLv3",
    keywords = "sar, feature tracking, ice drift",
    url = "https://github.com/nansencenter/sea_ice_drift",
    packages=['sea_ice_drift'],
    package_data={'sea_ice_drift': ['sea_ice_drift/testdata/*']},
    install_requires=['nansat', 'cv2'],
    test_suite="sea_ice_drift.tests",
)
