#!/bin/bash

# $1 - hosts file
# $2 - key

hosts=$1
key=$2

central_ip="$(hostname -I | awk '{print $1}')"
echo "Running bandwidth all from $central_ip"

base_port=5000
while read ip_addr; do
    ssh -n -i "$key" -o StrictHostKeyChecking=no ubuntu@"$ip_addr" \
        "cd $PWD; ./bandwidth-one-to-all.sh $central_ip $hosts $key $base_port; exit;" &
    base_port=$(($base_port + 1000))
done < $hosts

wait

echo "src_ip,dst_ip,bandwidth" > data/bandwidth_data.csv
cat data/bandwidth-*.csv >> data/bandwidth_data.csv

