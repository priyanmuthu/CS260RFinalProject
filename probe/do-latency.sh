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
latency_outfile="$PWD/data/latency-$src_ip-$dst_ip.tmp"
rm -f "$latency_outfile" >/dev/null 2>&1 && touch "$latency_outfile"

# Run latency probe
echo "    Probing latency from $src_ip to $dst_ip on port $port"
ssh -n -i "$key" -o StrictHostKeyChecking=no ubuntu@"$dst_ip" \
    "cd $PWD; sockperf ping-pong --ip $src_ip --port $port --time 3; exit;" > "$latency_outfile"
latency="$(cat "$latency_outfile" | sed -n 's/^.*Summary: Latency is \(.*\) usec/\1/p')"

# Reformat output files
echo "$src_ip,$dst_ip,$latency" > "$latency_outfile"

