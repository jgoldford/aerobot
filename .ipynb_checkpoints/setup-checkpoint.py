from setuptools import find_packages, setup

setup(
	name = "AEROBOT",
	version = '0.1',
	description = 'Python package for predicting oxygen requirements from prokaryotic genomes',
	url = 'https://github.com/jgoldford/aerobot',
	author = 'Joshua E. Goldford, Avi Flamholz',
	author_email = 'goldford.joshua@gmail.com',
	packages = find_packages(),
	install_requires = [],
	include_package_data = True,
)