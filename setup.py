from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="solat_cb",
    version="1.0",
    packages=find_packages(include=['solat_cb', 'solat_cb.*']),
    include_package_data=True,
    description="A package for  CB analysis with SO LAT",
    maintainer="Anto I. Lonappan",
    maintainer_email="mail@antolonappan.me", 
)
 