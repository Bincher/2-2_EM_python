import math
g = 0
m = 0
c = 1
t = 0

def v(t):
    return math.sqrt((g * m)/c) * math.tanh(math.sqrt(((g * c) / m) * t))

