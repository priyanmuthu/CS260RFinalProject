#!/bin/bash

# $1 - central IP
# $2 - hosts
# $3 - key
# $4 - base port

central_ip=$1
hosts=$2
key=$3
base_port=$4

# Find this IP address
src_ip="$(hostname -I | awk '{print $1}')"
echo "  Probing bandwidth from $src_ip to all"

# Prepare output files
mkdir -p "$PWD/data/"
bandwidth_outfile="$PWD/data/bandwidth-$src_ip.csv"

# Process only hosts including and after `src_ip`
cat $hosts | sed -n '/'"$src_ip"'$/,$p' | sed 1d > new_hosts.tmp
# If this is the last host, nothing to do
if [ ! -s new_hosts.tmp ]; then
    rm -f new_hosts.tmp && exit
fi

# Probe all for bandwidth
running_port=$base_port
while read ip_addr; do
    running_port=$(($running_port + 1))
    sockperf server --ip $src_ip --port $running_port > /dev/null 2>&1 &
    ./do-bandwidth.sh $src_ip $ip_addr $running_port $key &
done < new_hosts.tmp
wait $(pgrep do-bandwidth)
kill -s SIGINT $(pgrep sockperf)
rm -f new_hosts.tmp

cat data/bandwidth-*.tmp > "$bandwidth_outfile"
rm -f data/bandwidth-*.tmp

echo "  Copying results to $central_ip"
scp -i "$key" -o StrictHostKeyChecking=no "$bandwidth_outfile" \
    ubuntu@"$central_ip":"$PWD/data/"

