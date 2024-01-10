import sys


# Original string
old_intor1 = '(gen-cint "intor1.c"'


# Patch string
new_intor1 = old_intor1


# Momentum
new_intor1 += '\n  \'("int1e_p" ( \\| p))'


# Angular momentum
new_intor1 += '\n  \'("int1e_rxp" ( \\| rc cross p))'


# Moment components
for nx in range(5):
    for ny in range(5):
        for nz in range(5):
            # z and zz are already in ``libcint``
            if 0 < nx + ny + nz < 5 and (nx, ny, nz) != (0, 0, 1) and (nx, ny, nz) != (0, 0, 2):
                name = nx * "x" + ny * "y" + nz * "z"
                oper = nx * "xc " + " " + ny * "yc " + " " + nz * "zc "
                new_intor1 += f'\n  \'("int1e_{name}" ( \\| {oper} \\| ))'


# Read script
with open(sys.argv[1], 'r') as f:
    txt = f.read()


# Patch script
with open(sys.argv[1], 'w') as f:
    f.write(txt.replace(old_intor1, new_intor1))
