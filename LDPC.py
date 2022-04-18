import argparse
import numpy as np
from pyldpc import make_ldpc, encode, decode, get_message
import random

parser = argparse.ArgumentParser(description='LDPC encoder/decoder', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-i', '--input', help='input msg to use. defaults to random.')
parser.add_argument('-e', '--encode', help='Do encoding', action='store_true')
parser.add_argument('-p', '--pyldpc', help='Do encoding with pyldpc', action='store_true')
parser.add_argument('-d', '--decode', help='Do decoding', action='store_true')
parser.add_argument('-f', '--errors', help='Errorcount. default: 0', type=int)
parser.add_argument('-s', '--silent', help='Dont print', action='store_true')
parser.add_argument('-w', '--wait', help='Do decoding step by step and wait for input.', action='store_true')
parser.add_argument('-H', '--matrix', help='WIP: parity matrix n,m')

H_ = [
    [0, 1, 0, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 1, 0]
]
H__ = [  # Example 6.7 from "A Practical Guide to Error-Control Coding Using MATLAB" // FIXED SECOND TO LAST BIT.
    [1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 1]
]
H = [
    [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0]
]

checker_nodes = []
known_c = []  # Flags for known checker nodes.

v_nodes = []
known_v = []  # Flags for known v nodes.


def main():
    # TODO: remove global lists used in encoding. Use numpy for better matrix operations...
    global H
    global v_nodes
    code = []
    args = parser.parse_args()

    if args.matrix:
        # TODO. This doesn't work at all.
        dimensions = args.matrix.split(",")
        n = int(dimensions[0])
        m = int(dimensions[1])
        H = generate_matrix(n, m, n / 2, m / 2)

    if args.input:
        for s in args.input:
            if s in ["0", "1"]:
                code.append(int(s))
            else:
                print("Please input proper code. " + s + " != 0 or 1")
    else:
        # if no input is given, make random input. Probably won't work with the small H matrix...
        i_len = len(H[0])
        if args.encode:
            i_len -= int(i_len / 4)  # leave some bits "open" for encoding to do stuff.. (Not very polished..)
        for i in range(i_len):
            code.append(random.randint(0, 1))

    ercount = 0
    if args.errors is not None:
        ercount = args.errors
    if not args.silent and not args.pyldpc:
        print("Input: " + intlist2str(code))
        print("Matrix: H =")
        for row in H:
            print("  " + intlist2str(row))

    if args.encode:
        custom_encode(code)
        if not args.silent:
            print("Checker nodes: " + intlist2str(checker_nodes))
            print("Encodign result: " + intlist2str(v_nodes))
    else:
        v_nodes = code[:]

    msg = v_nodes[:]

    if args.pyldpc:
        # Generate the matrix and message with pyldpc
        n = 15
        d_v = 4
        d_c = 5
        snr = 20
        H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
        k = G.shape[1]
        v = np.random.randint(2, size=k)
        y = encode(G, v, snr)
        # hacky: y has soft values so decode it with pyldpc to get hard values...
        d = decode(H, y, snr)
        msg = d[:]
        print("Message: " + intlist2str(msg))
        print("Matrix: H =")
        for row in H:
            print("  " + intlist2str(row))

    if ercount > 0:
        msg = add_errors(msg, ercount)
        if not args.silent:
            print("Msg after added (" + str(ercount) + ") errors: " + intlist2str(msg))
    if args.decode:
        msg = custom_decode(msg, args.wait)
        if type(msg) == type(False):
            if not args.silent:
                print("Could not decode the message")
            return False
        else:
            if not args.silent:
                print("Decoding result: " + intlist2str(msg))
            if ercount > 0 and args.encode:  # If message was encoded/decoded check if it matches with input.
                success = True
                for i, b in enumerate(msg):
                    if b != v_nodes[i]:
                        success = False
                if success:
                    if not args.silent:
                        print("Message correctly decoded")
                    return msg
                else:
                    if not args.silent:
                        print("Error in decoding the message")
                    return False
    return msg


def custom_encode(code):
    # This could be done by just multiplying the input msg with Generator matrix G which you get from the H with some matrix magic.
    # TODO: matrix magic for the generator matrix.
    init_nodes(code)
    set_parity_bits()


def init_nodes(code):
    for f in H:
        checker_nodes.append(0)
        known_c.append(0)
    for c in range(len(H[0])):
        if c < len(code):  # bits in code already known.
            v_nodes.append(code[c])
            known_v.append(1)
        else:
            known_v.append(0)
            v_nodes.append(0)


def set_parity_bits():
    while 0 in known_v:
        change = False  # flag for detecting infinite loops, just in case
        for i in range(len(checker_nodes)):
            if known_c[i]:
                continue  # Checker node already complete, do next
            unknown = -1  # unknown bits for checker node
            parity = 0
            for j in range(len(v_nodes)):
                if H[i][j]:
                    if not known_v[j]:
                        if unknown != -1:  # more than 1 unknown. skip checker node
                            unknown = -1
                            break
                        unknown = j
                    else:
                        parity ^= v_nodes[j]
            if unknown != -1:
                checker_nodes[i] = parity
                v_nodes[unknown] = parity
                known_v[unknown] = 1
                known_c[i] = 1
                change = True  # progress
        if not change:  # should not ever get here
            print("Could not compute parity bits")
            break


def add_errors(msg, errcount):
    for i in range(errcount):
        er = random.randint(0, len(msg) - 1)
        msg[er] ^= 1
    return msg


def decode2(msg, wait):
    loop = 0
    while True:
        loop += 1
        out = msg[:]
        satisfied = True
        c_nodes = []
        for i in range(len(H)):
            c_nodes.append(0)
        for i in range(len(H)):
            for j in range(len(H[i])):
                if H[i][j]:
                    c_nodes[i] ^= msg[j]
            for j in range(len(H[i])):
                if H[i][j]:
                    out[j] += msg[j] ^ c_nodes[i]
            if c_nodes[i]:
                satisfied = False
        if satisfied:
            break
        for j in range(len(out)):
            div = 1
            for i in range(len(H)):
                div += H[i][j]
            v = out[j] / div
            if v > 0.5:  # Not sure if this should be >= instead. or if there should be some other logic when its even choice..
                out[j] = 1
            else:
                out[j] = 0
        msg = out[:]
        if wait:
            if loop == 1:
                print("(Press enter to continue decoding)")
            input(str(loop) + ": " + intlist2str(msg))
        if loop > 10000:  # Hardcoded max loop values
            return False
    return msg


def custom_decode(msg, wait):
    loop = 0
    while True:
        loop += 1
        c_nodes = []
        failcount = []
        for i in range(len(H)):
            c_nodes.append(0)
        for i in range(len(H[0])):
            failcount.append(0)
        for i in range(len(H)):
            for j in range(len(H[i])):
                if H[i][j]:
                    c_nodes[i] ^= msg[j]
        for i in range(len(H)):
            for j in range(len(H[i])):
                if H[i][j]:
                    if c_nodes[i]:
                        failcount[j] += 1
        most_fails = 0
        fail_index = 0
        for i, f in enumerate(failcount):
            if f > most_fails:
                most_fails = f
                fail_index = i
        if most_fails == 0:
            break
        msg[fail_index] ^= 1
        if wait:
            if loop == 1:
                print("(Press enter to continue decoding)")
            print(str(loop - 1) + ": " + intlist2str(failcount) + " # of failed parity checks")
            indicator = list(" " * len(failcount))
            indicator[fail_index] = "^"
            print(str(loop - 1) + ": " + intlist2str(indicator))
            input(str(loop) + ": " + intlist2str(msg))
        if loop > 10000:  # Hardcoded max loop values
            return False
    return msg


def intlist2str(intlist):
    return "".join([str(bit) for bit in intlist])


if __name__ == '__main__':
    main()
