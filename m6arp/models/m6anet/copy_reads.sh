#!/usr/bin/env bash

WORK_DIR=$1
READS_FILE=$2
DESTINATION=$3

cd $WORK_DIR

mkdir -p $DESTINATION

for READ in $(cat $READS_FILE); do
cp $READ $DESTINATION
done
