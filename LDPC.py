import argparse
import math

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

H___ = [
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
H = [
    [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
]

checker_nodes = []
known_c = []  # Flags for known checker nodes.

v_nodes = []
known_v = []  # Flags for known v nodes.


def main(runtimes=1):
    # TODO: remove global lists used in encoding. Use numpy for better matrix operations...
    global H
    global v_nodes
    code = []
    args = parser.parse_args()

    if args.matrix:
        pass
        # TODO. This doesn't work at all.
        # dimensions = args.matrix.split(",")
        # n = int(dimensions[0])
        # m = int(dimensions[1])
        # H = generate_matrix(n, m, n / 2, m / 2)

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
            print("Encoding result: " + intlist2str(v_nodes))
    else:
        v_nodes = code[:]

    msg = v_nodes[:]

    if args.pyldpc:
        # Generate the matrix and message with pyldpc
        win = 0
        sigmawin = 0
        for x in range(runtimes): #Quick way to test multiple times.
            n = 15
            d_v = 2
            d_c = 5
            snr = 5
            seed = random.randint(0, 2**32-1)
            H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)
            k = G.shape[1]
            v = np.random.randint(2, size=k)

            y = encode(G, v, snr, seed=seed)
            # pyLDPC encoder gives us soft values with errors.
            d = decode(H, y, snr)
            msg = d[:]

            # TODO: move all code bellow to the decoding segment..
            for sigma in [1, 0.9, 1.1, 0.8, 1.2, 0.6, 1.4, 0.3, 1.7]:
                output = spd(y, sigma)
                print("og-viesti: " + intlist2str(v))
                print("slothldpc: " + intlist2str(output))
                print("pyldpc:    " + intlist2str(msg))
                if intlist2str(output) == intlist2str(msg):
                    print(f"{x}: Result ok!")
                    if sigma != 1:
                        print(f"Fixed by fiddling with sigma: {sigma}")
                        sigmawin += 1
                    else:
                        win += 1
                    break
                else:
                    errors = 0
                    for i, b in enumerate(output):
                        if b != msg[i]:
                            errors += 1
                    print(f"{x}: Error: {errors}")
        print(f"Finished: {win} correct results out of 500. {sigmawin} were fixed by fiddling with sigma value")

    if ercount > 0:
        msg = add_errors(msg, ercount)
        if not args.silent:
            print("Msg after added (" + str(ercount) + ") errors: " + intlist2str(msg))
    if args.decode:
        #Hard decision decoding
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


def spd(msg, sigma):
    print(msg)
    print("---")
    print(H)
    p0 = []
    p1 = []
    pp0 = []
    pp1 = []
    for r in msg:
        j = 1 / (1 + math.exp((2 * r) / sigma))
        p1.append(j)
        p0.append(1 - j)

    og_mesg = get_bits(p0, p1)
    #print("withnoice: " + intlist2str(og_mesg))
    for i, row in enumerate(H):
        pp1i = []
        pp0i = []
        for j, c in enumerate(row):
            pp1i.append(c * p1[j])
            pp0i.append(c * p0[j])
        pp0.append(pp0i)
        pp1.append(pp1i)
    oldbits = og_mesg[:]
    oldbits2 = og_mesg[:]
    oldbits3 = og_mesg[:]
    max_iter = 1000
    for x in range(max_iter):
        # 1.
        deltapp = get_deltapp(pp1, pp0)
        deltaQ, Q0, Q1 = get_deltaQ(deltapp)

        # print_floatlistlist(deltaQ)
        # print_floatlistlist(Q0)
        # print_floatlistlist(Q1)
        # 2.
        pp0, p0 = get_pp(p0, Q0)
        pp1, p1 = get_pp(p1, Q1)
        pp0, pp1 = scale_pp(pp0, pp1)
        p0, p1 = scale_p(p0, p1)
        # print_floatlistlist(pp0)
        # print_floatlistlist(pp1)
        # 3.
        bits = get_bits(p0, p1)
        if oldbits == bits and oldbits2 == bits and oldbits3 == bits:
            print("ready in: " + str(x))
            break
        oldbits3 = oldbits2[:]
        oldbits2 = oldbits[:]
        oldbits = bits[:]
        if x == max_iter -1:
            print("Reached max iterations")

    errors = 0
    for i, b in enumerate(bits):
        if b != og_mesg[i]:
            errors += 1
    print(f"fixed {errors} errors!")  # thank runtu for string formating help
    return bits


def get_bits(p0, p1):
    bits = []
    for i in range(len(p0)):
        if p0[i] > p1[i]:
            bits.append(0)
        else:
            bits.append(1)
    return bits


def get_deltapp(pp1, pp0):
    deltapp = []
    for i, row in enumerate(pp1):
        deltappi = []
        for j, c in enumerate(row):
            deltappi.append(pp0[i][j] - pp1[i][j])
        deltapp.append(deltappi)
    return deltapp


def get_deltaQ(deltapp):
    deltaQ = []
    Q0 = []
    Q1 = []
    for i, row in enumerate(deltapp):
        deltaQi = []
        Q0i = []
        Q1i = []
        for j, c in enumerate(row):
            product = 1
            if c != 0:
                for j2, c2 in enumerate(row):
                    if j2 != j and c2 != 0:
                        product = product * c2
                deltaQi.append(product)
                Q0i.append(0.5 * (1 + product))
                Q1i.append(0.5 * (1 - product))
            else:
                deltaQi.append(0)
                Q0i.append(0)
                Q1i.append(0)
        deltaQ.append(deltaQi)
        Q0.append(Q0i)
        Q1.append(Q1i)
    return deltaQ, Q0, Q1


def get_pp(inp, inQ):
    pp = []
    p = []
    for z in inQ[0]:
        p.append(0)
    for i, row in enumerate(inQ):
        ppj = []
        for j, c in enumerate(row):
            product = 1
            product2 = 1
            if c != 0:
                for i2 in range(len(inQ)):
                    if inQ[i2][j] != 0:
                        product2 = product2 * inQ[i2][j]
                        if i2 != i:
                            product = product * inQ[i2][j]
                p[j] = inp[j] * product2
                ppj.append(inp[j] * product)
            else:
                ppj.append(0)
        pp.append(ppj)
    return pp, p


def scale_pp(pp0, pp1):
    for j in range(len(pp0)):
        for i in range(len(pp0[j])):
            if pp0[j][i] + pp1[j][i] != 0:
                factor = 1 / (pp0[j][i] + pp1[j][i])
                pp0[j][i] *= factor
                pp1[j][i] *= factor
    return pp0, pp1


def scale_p(p0, p1):
    for i in range(len(p0)):
        if p0[i] == "?":
            print(p0)
        if p1[i] == "?":
            print(p1)
        if p0[i] + p1[i] != 0:
            factor = 1 / (p0[i] + p1[i])
            p0[i] *= factor
            p1[i] *= factor
    return p0, p1


def print_floatlistlist(l):
    for j in l:
        fl = []
        for i in j:
            f = "%.2f" % i
            fl.append(f)
        print(fl)



def logspd(msg, sigma):
    #init
    pi = []
    og_mesg = []
    for r in msg:
        j = (2 * r) / sigma
        pi.append(j)
        if j < 0:
            og_mesg.append(1)
        else:
            og_mesg.append(0)
    print(pi)
    oldbits = og_mesg[:]
    oldbits2 = og_mesg[:]
    oldbits3 = og_mesg[:]

    p = []
    for j in range(len(H)):
        prow = []
        for i, c in enumerate(pi):
            prow.append(H[j][i]*c)
        p.append(prow)
    # 1.

    max_iter = 1000
    for x in range(max_iter):

        Q = []
        for j, row in enumerate(p):
            Qrow = []
            for i, col in enumerate(row):
                if H[j][i] == 0:
                    Qrow.append(0)
                    continue
                summing = 0
                for j2, c in enumerate(p):
                    if j == j2:
                        continue
                    summing += theta(abs(c[i]))
                summing = theta(summing)
                product = 1
                for j2, c in enumerate(p):
                    if j == j2:
                        continue
                    product *= sign(c[i]) * summing
                Qrow.append(product)
            Q.append(Qrow)

        # 2.
        newP = []
        for j, row in enumerate(p):
            Prow = []
            for i, col in enumerate(row):
                if H[j][i] == 0:
                    Prow.append(0)
                    continue
                summing = 0
                for i2, c in enumerate(row):
                    if i == i2:
                        continue
                    summing += Q[j][i]
                Prow.append(p[j][i]+summing)
            newP.append(Prow)
        p = newP[:]
        # 3.
        res = []
        for i in p[0]:
            res.append(0)
        bits = res[:]
        for j, row in enumerate(p):
            for i, col in enumerate(row):
                summing = 0
                for i2, c in enumerate(row):
                    summing += Q[j][i]
                val = p[j][i]+summing
                if val != 0:
                    res[i] = val
                    if val < 0:
                        bits[i] = 1
        print(res)
        if oldbits == bits and oldbits2 == bits and oldbits3 == bits:
            print("ready in: " + str(x))
            break
        oldbits3 = oldbits2[:]
        oldbits2 = oldbits[:]
        oldbits = bits[:]
        if x == max_iter -1:
            print("Reached max iterations")

    errors = 0
    for i, b in enumerate(bits):
        if b != og_mesg[i]:
            errors += 1
    print(f"fixed {errors} errors!")
    return bits


def theta(val):
    if val == 0:
        return 0
    return -math.log(math.tanh(val/2))


def sign(val):
    if val < 0:
        return -1
    else:
        return 1

if __name__ == '__main__':
    main(1)
