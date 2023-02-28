from pyldpc import make_ldpc, encode, decode, get_message
import numpy as np
import random

testcases = 1

sizes = []
sizenames = {}
cvars = {}
prenames = ["OGMESLEN", "MSGLEN", "HROWS"]

filename = "CVARS.txt"

doprint = True
dofile = True

autoname = True

def main():
    # Generate the matrix and message with pyldpc
    for x in range(testcases): #Quick way to test multiple times.
        n = 50
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

        getascvar(v, "input", x)
        getascvar(y, "message", x)
        getascvar(msg, "reference", x)
        getascvar(H, "h_matrix", x)

    namesizes()
    #printcvars()

def getascvar(v,varname, set):
    vtype = ""
    vsize = []
    if len(v) > 0:
        if type(v) is np.ndarray:
            vsize.append(len(v))
            if type(v[0]) is np.ndarray:
                vsize.append(len(v[0]))
                if type(v[0][0]) is np.int64:
                    vtype = "int"
                if type(v[0][0]) is np.float64:
                    vtype = "float"
            if type(v[0]) is np.int64:
                vtype = "int"
            if type(v[0]) is np.float64:
                vtype = "float"

    value = ""
    if type(v) is np.ndarray:
        value = "{"
        if type(v[0]) is np.ndarray:
            rows = []
            for row in v:
                r = "{"
                r += ", ".join(map(str, row))
                r += "}"
                rows.append(r)
            value += ", ".join(rows)
        else:
            value += ", ".join(map(str, v))
        value += "}"
    else:
        value = str(v)
    for s in vsize:
        if s not in sizes:
            sizes.append(s)
    if set not in cvars.keys():
        cvars[set] = []
    cvars[set].append({"t": vtype, "n": varname, "s": vsize[:], "v": value})

def namesizes():
    if not autoname:
        print("Found " + str(len(sizes)) + " sizes:")
        sline = ""
        for i, s in enumerate(sizes):
            if len(sizes) == len(prenames):
                sline += '"' + prenames[i] + '": ' + str(s) + "  "
            else:
                sline += str(s) + "  "
        print(sline)
        rn = input("Rename? [y/N]")
    else:
        rn = ""

    for i, s in enumerate(sizes):
        if rn == "y":
            inp = input(str(s) + ": ")
            if inp == "":
                sizenames[s] = s
            else:
                sizenames[s] = inp
        elif len(sizes) == len(prenames):
                sizenames[s] = prenames[i]
        else:
            sizenames[s] = s
    if not autoname:
        print("\n")


def printcvars():
    for k in sizenames.keys():
        if k != sizenames[k]:
            dowrite("#define " + sizenames[k] + " " + str(k))
    dowrite("#define TESTCASECOUNT " + str(testcases))

    dowrite("")

    dowrite("typedef struct Testcase {")
    for v in cvars[0]:
        line = "    " + v["t"] + " " + v["n"]
        for s in v["s"]:
            line += "[" + sizenames[s] + "]"
        line += ";"
        dowrite(line)
    dowrite("} Testcase;")
    for set in cvars.keys():
        dowrite("\nstruct Testcase case_" + str(set+1) + " = {")
        for v in cvars[set]:
            dowrite("    " + v["v"]+",")
        dowrite("};")

    dowrite("\nTestcase * cases[TESTCASECOUNT] = {")
    for set in cvars.keys():
        dowrite("    &case_" + str(set+1) +",")
    dowrite("};")

    if dofile:
        write_to_file()

file = ""
def dowrite(s):
    global file
    if doprint:
        print(s)
    if dofile:
        file += s+"\n"

def write_to_file():
    f = open(filename, "w")
    f.write(file)
    f.close()


if __name__ == '__main__':
    main()
