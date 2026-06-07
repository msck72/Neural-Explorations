from inference_tensor_cpp import _InferenceTensor

class InferenceTensor(_InferenceTensor):
    def __getitem__(self, key):
        if not isinstance(key, (list, tuple)):
            key = [key]
        return self.get_item(list(key))

    def __setitem__(self, key, value):
        if not isinstance(key, (list, tuple)):
            key = [key]
        self.set_item(list(key), value)
    
    def __matmul__(self, other):
        res = _InferenceTensor.matmul(self, other)
        out = InferenceTensor(list(res.shape))
        _InferenceTensor.set_values(out, res.data)
        return out

    def __add__(self, other):
        res = _InferenceTensor.__add__(self, other)
        out = InferenceTensor(list(res.shape))
        _InferenceTensor.set_values(out, res.data)
        return out

    def __sub__(self, other):
        res = _InferenceTensor.__sub__(self, other)
        out = InferenceTensor(list(res.shape))
        _InferenceTensor.set_values(out, res.data)
        return out

    def __mul__(self, other):
        res = _InferenceTensor.__mul__(self, other)
        out = InferenceTensor(list(res.shape))
        _InferenceTensor.set_values(out, res.data)
        return out
    
    def __repr__(self):
        # print("Calling __repr__")
        self.print()
        return ""

    def set_values(self, values):
        # print(f'self.shape: {self.shape}, values shape: {values.shape}')
        if values.shape != tuple(self.shape):
            raise ValueError(f"Shape mismatch: expected {self.shape}, got {values.shape}")
        
        super().set_values(values.flatten())