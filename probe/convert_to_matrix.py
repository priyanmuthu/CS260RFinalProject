import sys
from csv import DictReader

# Returns a tuple of two dictionaries
# The first one is a dictionary of average latencies
# The second one is a dictionary of standard deviations
def get_matrix(filename):
    matrix_avg = {}
    matrix_sdev = {}
    with open(filename, "r") as infile:
        reader = DictReader(infile)
        for row in reader:
            src_ip = row['src_ip']
            dst_ip = row['dst_ip']
            matrix_avg[(src_ip, dst_ip)] = row['avg']
            # This is standard deviation, but `ping` calls it mdev
            # No idea why
            matrix_sdev[(srx_ip, dst_ip)] = row['mdev']

    return matrix_avg, matrix_sdev


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python %s filename" % sys.argv[0])
    else:
        matrix_avg, matrix_sdev = get_matrix(sys.argv[1])
