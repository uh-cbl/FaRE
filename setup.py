from __future__ import absolute_import
from setuptools import setup, find_packages

kwargs = {'install_requires': ['numpy', 'opencv-python', 'scikit-learn', 'scikit-image', 'matplotlib', 'seaborn',
                               'scipy', 'tqdm'],
          'package_data': {'fare.datasets': ['**/*.txt', '**/**/**/**/*.txt']}}

setup(
    name='fare',
    version='0.2.1',
    packages=find_packages(),
    include_package_data=True,
    url='https://github.com/uh-cbl/FaRE/',
    license='Copyright@Computational Biomecidine Lab',
    author='Xiang Xu',
    author_email='xxu21@uh.edu',
    description='The python library for evaluating the face recognition performance',
    **kwargs
)
