#!/bin/bash

# $1 - hosts file
# $2 - key

hosts=$1
key=$2

./latency-all.sh $hosts $key
./bandwidth-all.sh $hosts $key

