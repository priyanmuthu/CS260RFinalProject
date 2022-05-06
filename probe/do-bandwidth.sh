#!/bin/bash

# $1 - src ip
# $2 - dst ip
# $3 - port
# $4 - key

src_ip=$1
dst_ip=$2
port=$3
key=$4

# Prepare output file
bandwidth_outfile="$PWD/data/bandwidth-$src_ip-$dst_ip.tmp"
rm -f "$bandwidth_outfile" >/dev/null 2>&1 && touch "$bandwidth_outfile"

# Run bandwidth probe
echo "    Probing bandwidth from $src_ip to $dst_ip on port $port"
ssh -n -i "$key" -o StrictHostKeyChecking=no ubuntu@"$dst_ip" \
    "cd $PWD; sockperf throughput --ip $src_ip --port $port --time 3; exit;" > "$bandwidth_outfile"
bandwidth="$(cat "$bandwidth_outfile" | sed -n 's/^.*Summary: BandWidth is .* MBps (\(.*\) Mbps)/\1/p')"

# Reformat output files
echo "$src_ip,$dst_ip,$bandwidth" > "$bandwidth_outfile"

