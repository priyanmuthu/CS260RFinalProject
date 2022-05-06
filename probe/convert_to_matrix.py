import sys
from csv import DictReader


def parse(file_name, entry_name):
    assert entry_name == "latency" or entry_name == "bandwidth", \
           "Can parse only latencies or bandwidths"
    matrix = {}
    with open(file_name, "r") as infile:
        reader = DictReader(infile)
        for row in reader:
            src_ip = row['src_ip']
            dst_ip = row['dst_ip']
            # The matrix is symmetric
            matrix[(src_ip, dst_ip)] = matrix[(dst_ip, src_ip)] = \
                float(row[entry_name])
            # Latency from a node to itself is 0, bandwidth is infinite
            matrix[(src_ip, src_ip)] = matrix[(dst_ip, dst_ip)] = \
                float("inf") if entry_name == "bandwidth" else 0.0

    return matrix


# Returns a dictionary with latencies
# For example, if we had an ip 172.31.1.233 and an ip 172.31.9.249,
# then we have
#     matrix[("172.31.1.233", "172.31.9.249")] == matrix[("172.31.9.249", "172.31.1.233")]
def get_latencies(file_name):
    return parse(file_name, "latency")


# Returns a dictionary with bandwidths, works same as latencies
def get_bandwidths(file_name):
    return parse(file_name, "bandwidth")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python %s latency_filename bandwidth_filename" % sys.argv[0])
    else:
        latencies = get_latencies(sys.argv[1])
        bandwidths = get_bandwidths(sys.argv[2])
        for ip_pair, latency in latencies.items():
            if ip_pair[0] == ip_pair[1]:
                assert latency == 0.0
            else:
                assert isinstance(latency, float)
                assert latency == latencies[(ip_pair[1], ip_pair[0])]

        for ip_pair, bandwidth in bandwidths.items():
            if ip_pair[0] == ip_pair[1]:
                assert bandwidth == float("inf")
            else:
                assert isinstance(bandwidth, float)
                assert bandwidth == bandwidths[(ip_pair[1], ip_pair[0])]
