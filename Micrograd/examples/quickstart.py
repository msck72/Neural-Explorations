from Micrograd.engine import Value
from Micrograd.utils import draw_graph
a = Value(2.0)
b = Value(3.0)

c = a + b
d = c + 4
e = d / c

e.backward()

draw_graph(e)