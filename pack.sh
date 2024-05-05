#!/bin/bash

ZIP_NAME="xdvora3o.zip"

zip -r "$ZIP_NAME" data/* results src/* experiments/* readme.txt prezentace.pdf -x results/initial/**\* results/**/*.jpg

