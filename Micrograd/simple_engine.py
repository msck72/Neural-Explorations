import math
from utils import draw_dot

class SimpleValue:
    def __init__(self, data, _formed_by=(), _op=None):
        self.data = data
        self._formed_by=_formed_by
        self._op = _op
        self._backward = lambda : None
        self.grad = 0
        

    def __add__(self, other):
        if type(other) is not SimpleValue:
            other = SimpleValue(other)
        
        out = SimpleValue(self.data + other.data, _formed_by=(self, other), _op='+')

        def _backward():
            self.grad += out.grad 
            other.grad += out.grad

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        if type(other) is not SimpleValue:
            other = SimpleValue(other)

        out = SimpleValue(self.data * other.data, _formed_by=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * (other**(-1))

    def __pow__(self, n):
        if type(n) is not SimpleValue:
            n = SimpleValue(n)
        res = pow(self.data , n.data)
        out = SimpleValue(res, _formed_by=(self,), _op='**')
        
        def _backward():
            self.grad += (n.data - 1) * self.data * out.grad
            n.grad += (res * math.log(self.data)) * out.grad
        
        out._backward = _backward
        return out

    def tanh(self):
        e_x = SimpleValue(math.e) ** (self)
        e_neg_x = SimpleValue(math.e) ** (-1 * self)

        return (e_x - e_neg_x) / (e_x + e_neg_x)

    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._formed_by:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            print(v)
            v._backward()

    
    def __repr__(self):
        return f'data = {self.data} operation = {self._op}'
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-1 * other)
    
    def __rsub__(self, other):
        return (-1 * self) + other
    
    def __rtruediv__(self, other):
        return SimpleValue(other) / self