class Value:
    def __init__(self, data, formed_by=None):
        self.data = data
        self.formed_by=formed_by
        self._backward = lambda : None
        self.grad = 0
        

    def __add__(self, other):
        if type(other) is not Value:
            other = Value(other)
        
        out = Value(self.data + other.data, formed_by=(self, other))

        def _backward():
            self.grad += out.grad 
            other.grad += out.grad
            self._backward()
            other._backward()
        out._backward = _backward

        return out

    def __mul__(self, other):
        if type(other) is not Value:
            other = Value(other)

        out = Value(self.data * other.data, formed_by=(self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            self._backward()
            other._backward()
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        if type(other) is not Value:
            other = Value(other)
        
        return self * (other.pow(-1))

    def pow(self, n):
        res = pow(self.data , n)
        out = Value(res, formed_by=(self,))
        
        def _backward():
            self.grad += (n - 1) * self.data
            self._backward()
        
        out._backward = _backward
        return out

    
    def backward(self):
        self.grad = 1
        self._backward()

    def zero_grad(self):
        self.grad = 0
        op1, op2 = self.formed_by
        if op1 is not None:
            op1.zero_grad()
        if op2 is not None:
            op2.zero_grad()