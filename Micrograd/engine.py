import math

class Value:
    def __init__(self, data, _formed_by=(), _op=None):
        self.data = data
        self._formed_by=_formed_by
        self._op = _op
        self._backward = lambda : None
        self.grad = 0
        

    def __add__(self, other):
        if type(other) is not Value:
            other = Value(other)
        
        out = Value(self.data + other.data, _formed_by=(self, other), _op='+')

        def _backward():
            self.grad += out.grad 
            other.grad += out.grad
        
        out._backward = _backward
        return out

    def __mul__(self, other):
        if type(other) is not Value:
            other = Value(other)

        out = Value(self.data * other.data, _formed_by=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        if type(other) is not Value:
            other = Value(other)
        
        out = Value(self.data / other.data, _formed_by=(self, other), _op='/')

        def _backward():
            self.grad += (1 / other.data) * out.grad
            other.grad += (-1 * self.data / (other.data * other.data)) * out.grad

        out._backward = _backward
        return out

    def __pow__(self, n):
        if type(n) is not Value:
            n = Value(n)
        res = pow(self.data , n.data)
        out = Value(res, _formed_by=(self,), _op='**')
        
        def _backward():
            self.grad += (n.data - 1) * self.data * out.grad
            n.grad += (res * math.log(self.data)) * out.grad
        
        out._backward = _backward
        return out

    def tanh(self):
        tanh_x = math.tanh(self.data)
        out = Value(tanh_x, _formed_by=(self, ), _op='tanh')
        
        def _backward():
            self.grad += out.grad * (1 - tanh_x * tanh_x)
        
        out._backward = _backward
        return out

    
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
        return Value(other) / self