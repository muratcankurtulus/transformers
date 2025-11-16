#!/bin/bash
# Add the S3 prefix
PREFIX="https://data.commoncrawl.org/"

# Read the first 10 lines from wet.paths and download them
head -n 1 wet.paths | while read line; do
  wget "$PREFIX$line"
done
