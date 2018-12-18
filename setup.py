#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import shutil
import sys
from os.path import dirname, join

from setuptools import find_packages, setup


def get_version(package):
    """
    Return package version as listed in `__version__` in `init.py`.
    """
    init_py = open(join(package, '__init__.py')).read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)


version = get_version('ssdmultibox')


if sys.argv[-1] == 'publish':
    if os.system("pip freeze | grep twine"):
        print("twine not installed.\nUse `pip install twine`.\nExiting.")
        sys.exit()
    os.system("python setup.py sdist bdist_wheel")
    os.system("twine upload dist/*")
    print("You probably want to also tag the version now:")
    print("  git tag -a %s -m 'version %s'" % (version, version))
    print("  git push --tags")
    shutil.rmtree('dist')
    shutil.rmtree('build')
    shutil.rmtree('ssdmultibox.egg-info')
    sys.exit()


setup(
    name='ssd-pytorch',
    version=version,
    url='https://github.com/aaronlelevier/ssd-pytorch',
    license='MIT',
    description='SSD Object Detection using PyTorch',
    author='Aaron Lelevier',
    author_email='aaron.lelevier@gmail.com',
    packages=find_packages(exclude=['tests*', 'notebooks*']),
    include_package_data=True,
    install_requires=[],
    python_requires=">=3.6",
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Machine Learning',
        'Framework :: PyTorch',
        'Framework :: PyTorch :: 0.4.1',
        'Intended Audience :: Software Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ]
)
