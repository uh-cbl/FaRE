#!/usr/bin/env bash

python setup.py install
rm -r fare.egg-info
rm -r dist
rm -r build
