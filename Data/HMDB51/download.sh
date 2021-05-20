#!/bin/bash

#HMDB51
wget --no-check-certificate http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
mkdir videos
unrar e hmdb51_org.rar ./videos
cd ./videos
for file in *.rar; do unrar x "$file"; rm -rf "$file"; done
cd ../
rm -rf hmdb51_org.rar