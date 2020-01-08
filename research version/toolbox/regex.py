import re

with open('toolbox/debug_loss.txt', 'r') as fp:
    x = fp.read()

string = re.findall(r"loss: .+\n", x)
with open('toolbox/debug_loss_text.txt', 'w') as fp:
    fp.writelines(string)
    