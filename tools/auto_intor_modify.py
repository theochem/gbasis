import sys


old = '(gen-cint "intor1.c"'


new = f'{old}'
new += '\n  \'("int1e_p" ( \\| p))'
new += '\n  \'("int1e_rxp" ( \\| r cross p))'


with open(sys.argv[1], 'r') as f:
    txt = f.read()


with open(sys.argv[1], 'w') as f:
    f.write(txt.replace(old, new))
