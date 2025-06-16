# Tesnor as a multidimensional array
import random
from pprint import pformat

class Tensor:
    def __init__(self, shape, formed_by=(), op=None):

        def _create_multi_dim_aray(shape):
            if len(shape) == 1:
                return [random.random() for _ in range(shape[0])]
            return [_create_multi_dim_aray(shape[1:]) for _ in range(shape[0])]
        
        self.tensor = _create_multi_dim_aray(shape)
        self.shape = shape
        self._formed_by = formed_by
        self._op = None

        def _create_grad(shape):
            if len(shape) == 1:
                return [0.0 for _ in range(shape[0])]
            return [_create_grad(shape[1:]) for _ in range(shape[0])]

        self.grad = _create_grad(shape)
        self._backward = lambda: None
    
    def from_values(self, tensor, formed_by=(), op=None):
        self.tensor = tensor
        self._formed_by = formed_by
        self._op = op

    def _set_grad_to_one(self):
        def _helper(tensor_part, shape):
            if len(shape) == 1:
                for i in range(len(tensor_part)):
                    tensor_part[i] = float(1)
                return
            return [_helper(tensor_part[i], shape[1:]) for i in range(shape[0])]
        _helper(self.grad, self.shape)
        
    
    def get_shape(self):
        return self.shape
    
    def __add__(self, other):
        if self.shape != other.get_shape():
            raise Exception('Shape does not match')
        out = Tensor((self.shape))
        out.from_values(tensor=self._apply_on_tensor_parts(self.tensor, other.tensor, lambda x, y: x + y),
                        formed_by=(self, other),
                        op='+')
        
        def _helper(grad, chained_grad, shape):
            if len(shape) == 1:
                for i in range(len(chained_grad)):
                    grad[i] = chained_grad[i]
                return
            for i in range(len(grad)):
                _helper(grad[i], chained_grad[i], shape[1:])
            return
        
        def _backward():
            # self.grad = out.grad
            # other.grad = out.grad
            _helper(self.grad, out.grad, self.shape)
            _helper(other.grad, out.grad, other.shape)

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        if self.shape != other.get_shape():
            raise Exception('Shape does not match')
        out = Tensor((self.shape))
        out.from_values(tensor=self._apply_on_tensor_parts(self.tensor, other.tensor, lambda x, y: x * y),
                        formed_by=(self, other),
                        op='*')

        
        def _helper(grad, other_val, chained_grad, shape):
            if len(shape) == 1:
                for i in range(len(chained_grad)):
                    grad[i] = other_val[i] * chained_grad[i]
                return
            for i in range(len(grad)):
                _helper(grad[i], other_val[i], chained_grad[i], shape[1:])
            return
        
        def _backward():
            # self.grad = out.grad
            # other.grad = out.grad
            _helper(self.grad, other.tensor, out.grad, self.shape)
            _helper(other.grad, self.tensor, out.grad, other.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        if self.shape[-1] != other.shape[0] or len(self.shape) != 2 or len(other.shape) != 2:
            raise Exception('self.shape[-1] does not match other.shape[0]')

        def _matmul(a, b, c):
            
            # c = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]

            for i in range(len(c)):
                for j in range(len(c[0])):
                    for k in range(len(a[0])):
                        c[i][j] += a[i][k] * b[k][j]
            return c
                    

        out = Tensor((self.shape[0], other.shape[1]))
        out_tensor = [[0 for _ in range(len(other.tensor[0]))] for _ in range(len(self.tensor))]
        out.from_values(tensor=_matmul(self.tensor, other.tensor, out_tensor),
                        formed_by=(self, other),
                        op='@')

        def _backward():
            _matmul(out.grad, other.tensor, self.grad)
            _matmul(self.tensor, out.grad, other.grad)
        
        out._backward = _backward
        return out
    
    def _apply_on_tensor_parts(self, tensor_part, other_tensor_part, operation):
        if type(tensor_part) is float:
            return operation(tensor_part, other_tensor_part)
        return [self._apply_on_tensor_parts(tp, o_tp, operation) for tp, o_tp in zip(tensor_part, other_tensor_part)]

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

        print(f'len(topo) = {len(topo)}')
        for v in reversed(topo):
            print(f'v = {v}\n\n')
            
        # print(topo)
        self._set_grad_to_one()
        for v in reversed(topo):
            # print(f'v = {v}\n\n')
            v._backward()


    def __repr__(self):
        return pformat(self.tensor)
        # return self._get_tensor_str(self.tensor)
    
    def _get_tensor_str(self, tensor_part):
        if type(tensor_part) is float:
            return f'{tensor_part:.4f}'
        ans = '['
        for i in tensor_part:
            ans += f' {self._get_tensor_str(i)}'
        return ans + '],\n'

