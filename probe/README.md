# Probing

## Requirements

**Each node** needs to contain the following:

1. `sudo apt-get install sockperf` (for network measurements)
2. `python3` (for post-processing; if we're using some other language than Python then that's what we need to have)
3. A file that contains IP addresses of all nodes, such that each line is another IP address. Here is a sample file:
    ```
    172.31.1.233
    172.31.9.249
    172.31.5.128
	```
   No trailing newline! It doesn't matter how the file is named. Let's call it `hosts`.
4. A key for logging into each node. AWS requires the key to `ssh` into a machine. For example, if the key file is named `key.pem`, and we have a node with IP address `172.31.1.233`, we will `ssh` using
	```
	ssh -i key.pem ubuntu@172.31.1.233
	```
It might be possible to `ssh` without a key, but I did not look into this.

## How-To

Ensure that requirements are fulfilled. Then, run the following from *any* machine, does not matter which one:
```
./probe-all.sh hosts key.pem
```
where `hosts` is the name of the file that contains IP addresses of all nodes (see [Requirements](#requirements)), and `key.pem` is the name of the file that contains the key for logging into nodes (see [Requirements](#requirements)).

After the script runs, it will create a folder called `data`, with two relevant files: `latency_data.csv` and `bandwidth_data.csv`. The directory might also contain other files, but these are intermediate results, so not very relevant. Here's a sample `latency_data.csv`:
```
src_ip,dst_ip,latency
172.31.1.233,172.31.5.128,163.009
172.31.1.233,172.31.9.249,156.512
172.31.9.249,172.31.5.128,160.022
```
Note that the latency is expressed in **microseconds**. Here's a sample `bandwith_data.csv`:
```
src_ip,dst_ip,bandwidth
172.31.1.233,172.31.5.128,2.326
172.31.1.233,172.31.9.249,3.585
172.31.9.249,172.31.5.128,2.336
```
I made a Python script, `convert_to_matrix.py`, that parses these files into a convenient dictionary. Look at this to get an idea on how to use these `.csv` files.

## Notes

Currently, probing consists of two sequential steps. The first step is latency probing, and the second step is bandwidth probing. Within each of these steps, all operations run as concurrently as possible. We will see how that goes.

I picked `sockperf` because it was a single tool with a nice interface that could do both latency and bandwidth measurements. We are not committed to this decision, but from what I saw all these tools are more or less the same.

I did not test probing with many machines, so I don't know how long it takes. If it's too long, we can easily decrease the probing time. If you look at `do-latency.sh` and `do-bandwidth.sh`, the argument passed to `sockperf` is `--time 3`. This basically says "run the probe for 3 seconds." If this ends up being too long, we can reduce it.

