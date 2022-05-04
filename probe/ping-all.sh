#!/bin/bash

# $1 - hosts file
# $2 - frequency
# $3 - timeout
# $4 - key file

hosts=$1
freq=$2
timeout=$3
key=$4

central_ip="$(hostname -I | awk '{print $1'})"
echo "Running ping all from $central_ip"

while read line; do
    ssh -n -i "$key" -o StrictHostKeyChecking=no ubuntu@"$line" "cd $PWD; ./ping-one-to-all.sh $central_ip $hosts $freq $timeout $key" 
done < $hosts

echo "src_ip,dst_ip,min,avg,max,mdev" > data/ping-data.csv
cat data/ping-*.txt >> data/ping-data.csv

