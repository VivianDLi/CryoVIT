"""Setup script for CryoVIT."""

from setuptools import setup, find_packages


setup(
    name="cryovit",
    version="0.1.0",
    packages=find_packages(),  # Automatically find and include all packages
    package_data={"cryovit": ["configs/**/*"]},
    author="Sanket Rajan Gupte",
    author_email="sanketg@stanford.edu",
)
