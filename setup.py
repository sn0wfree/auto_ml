# coding=utf8
from setuptools import setup, find_packages
from core import __version__

setup(name='automl',
      version=__version__,
      url='https://github.com/sn0wfree/auto_ml',
      license='MIT',
      author='sn0wfree',
      author_email='snowfreedom0815@gmail.com',
      description='auto machine learning',
      packages=find_packages(exclude=['hyperopt.wiki']),
      long_description=open('README.md').read(),
      zip_safe=False,
      setup_requires=['hpsklearn>=0.0.3', 'hyperopt>=0.1.2', 'numpy>=1.14.3', 'requests>=0.9.1'],
      )
