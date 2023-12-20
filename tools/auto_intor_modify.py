import sys


old_intor1 = '(gen-cint "intor1.c"'


new_intor1 = f'{old_intor1}'
new_intor1 += "\n  ".join((
    # Momentum
    '\'("int1e_p" ( \\| p))',
    # Angular momentum
    '\'("int1e_rxp" ( \\| rc cross p))',
    # Moment X component
    '\'("int1e_x" ( \\| xc \\| ))',
    '\'("int1e_xx" ( \\| xc xc \\| ))',
    '\'("int1e_xxx" ( \\| xc xc xc \\| ))',
    '\'("int1e_xxxx" ( \\| xc xc xc xc \\| ))',
    # Moment Y component
    '\'("int1e_y" ( \\| yc \\| ))',
    '\'("int1e_yy" ( \\| yc yc \\| ))',
    '\'("int1e_yyy" ( \\| yc yc yc \\| ))',
    '\'("int1e_yyyy" ( \\| yc yc yc yc \\| ))',
    # Moment Z component
    #'\'("int1e_z" ( \\| zc \\| ))',             # already in `libcint`
    #'\'("int1e_zz" ( \\| zc zc \\| ))',         # already in `libcint`
    '\'("int1e_zzz" ( \\| zc zc zc \\| ))',
    '\'("int1e_zzzz" ( \\| zc zc zc zc \\| ))',
    ))


with open(sys.argv[1], 'r') as f:
    txt = f.read()


with open(sys.argv[1], 'w') as f:
    f.write(txt.replace(old_intor1, new_intor1))
