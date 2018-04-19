#!/bin/bash

DATASET=$1

# Catch abnormal curl behavior
if [ $? != 0 ]
then 
    echo Unable to retrieve URL, sorry!
    exit 1
fi


pdftotext $FILENAME.pdf

# Remove non ASCII characters 
LANG=C sed -i 's/[\d128-\d255]/\n/g' $FILENAME.txt

# I want to leave in tabs, commas, new lines,
# question marks, exclamations, and periods

# From the ascii table
# 0 - 8, 14 - 32, 34 - 43, 45, 47 - 62, 64, 
# 91 - 96, 123-127
LANG=C sed -i 's/[\d0-\d8]/\d10/g' $FILENAME.txt
LANG=C sed -i 's/[\d14-\d32]/\d10/g' $FILENAME.txt
LANG=C sed -i 's/[\d34-\d43]/\d10/g' $FILENAME.txt
LANG=C sed -i 's/\d45/\d10/g' $FILENAME.txt
LANG=C sed -i 's/[\d47-\d62]/\d10/g' $FILENAME.txt
LANG=C sed -i 's/\d64/\d10/g' $FILENAME.txt
LANG=C sed -i 's/[\d91-\d96]/\d10/g' $FILENAME.txt
LANG=C sed -i 's/[\d123-\d127]/\d10/g' $FILENAME.txt

# Delete all single characters by themselves
# (note- this may remove the word "I", but 
# can be modified)
#awk 'length!=1' $FILENAME.txt > ($FILENAME)_stripped.txt

# TODO decide if you want to operate on the stripped file
# or not. If you want to modify the file in place without
# creating a new file, rewrite as awk -i 'leng....

# Clean up after yourself
rm -rf $FILENAME.pdf
