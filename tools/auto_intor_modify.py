import sys


old = '(gen-cint "intor1.c"'


new = f'{old}\n'
new += '  \'("int1e_mom" (#C(-1 0) \\| p))'


with open(sys.argv[1], 'r') as f:
    txt = f.read()


with open(sys.argv[1], 'w') as f:
    f.write(txt.replace(old, new))
