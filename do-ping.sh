#!/bin/bash

# $1 - source IP
# $2 - destination IP
# $3 - output file
# $4 - frequency
# $5 - timeout

src_ip=$1
dst_ip=$2
outfile=$3
freq=$4
timeout=$5

echo "    Pinging from $src_ip to $dst_ip"

out="$(sudo ping $dst_ip -i $freq -w $timeout | tail -n 1)"
numbers=($(echo "$out" | sed -En 's/^.*= (.*) ms.*$/\1/p' | tr / \\n))

min=${numbers[0]}
avg=${numbers[1]}
max=${numbers[2]}
mdev=${numbers[3]}

echo "$src_ip,$dst_ip,$min,$avg,$max,$mdev" >> "$outfile"
