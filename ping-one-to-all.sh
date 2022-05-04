#!/bin/bash

# $1 - central IP
# $2 - hosts file
# $3 - frequency
# $4 - timeout
# $5 - key file

central_ip=$1
hosts=$2
freq=$3
timeout=$4
key=$5

src_ip="$(hostname -I | awk '{print $1'})"
echo "  Pinging from $src_ip to all"

rm -rf "$PWD/data/" >/dev/null 2>&1 && mkdir "$PWD/data/"
outfile="$PWD/data/ping-$src_ip.txt"
touch "$outfile"

while read line; do
    ./do-ping.sh "$src_ip" "$line" "$outfile" "$freq" "$timeout"
done < $hosts

echo "  Copying results to $central_ip"
scp -i "$key" -o StrictHostKeyChecking=no "$outfile" ubuntu@"$central_ip":"$outfile"

