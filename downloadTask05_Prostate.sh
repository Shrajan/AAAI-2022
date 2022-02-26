#!/bin/bash

mkdir data
wget -O train_and_test.zip https://zenodo.org/record/5500160/files/train_and_test.zip?download=1
unzip train_and_test.zip -d data/
mv data/MDCProstatePreprocessed/ data/train_and_test
rm train_and_test.zip
rm data/train_and_test/.DS_Store
rm -r data/__MACOSX/
