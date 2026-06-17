import numpy as np

def asdual(x):
    if isinstance(x, Dual):
        return x

    return Dual(x, 0.0)

def _add_dual(x: Dual, y: Dual):
    return Dual(x.a + y.a, x.b + y.b)

def _sub_dual(x: Dual, y: Dual):
    return Dual(x.a - y.a, x.b - y.b)

def _mul_dual(x: Dual, y: Dual):
    a, b = x.a, x.b
    c, d = y.a, y.b

    return Dual(a*c, a*d + b*c)

def _div_dual(x: Dual, y: Dual):
    a, b = x.a, x.b
    c, d = y.a, y.b
    det = b*c - a*d

    return Dual(a/c, det/c**2)

def _pow_dual(x: Dual, y: Dual):
    a, b = x.a, x.b
    c, d = y.a, y.b
    ac = a**c

    # handle case a == 0
    with np.errstate(divide='ignore', invalid='ignore'):
        b_out = ac * (d * np.log(a) + b*c / a)
        b_0 = np.where(c == 1, b, np.where(c > 1, 0, np.inf))

    return Dual(ac, np.where(a == 0, b_0, b_out))

def _eq_dual(x: Dual, y: Dual):
    return np.isclose(x.a, y.a)

def _neq_dual(x: Dual, y: Dual):
    return not np.isclose(x.a, y.a)

def _lt_dual(x: Dual, y: Dual):
    return x.a < y.a

def _leq_dual(x: Dual, y: Dual):
    return x.a <= y.a

def _gt_dual(x: Dual, y: Dual):
    return x.a > y.a

def _geq_dual(x: Dual, y: Dual):
    return x.a >= y.a

def _sin_dual(x: Dual):
    return Dual(np.sin(x.a), np.cos(x.a) * x.b)

def _cos_dual(x: Dual):
    return Dual(np.cos(x.a), -np.sin(x.a) * x.b)

def _exp_dual(x: Dual):
    ea = np.exp(x.a)
    return Dual(ea, ea * x.b)

def _log_dual(x: Dual):
    return Dual(np.log(x.a), x.b / x.a)

def _sqrt_dual(x: Dual):
    sa = np.sqrt(x.a)
    return Dual(sa, x.b / (2 * sa))

DUAL_BINARY_OPS = {
    np.add: _add_dual,
    np.subtract: _sub_dual,
    np.multiply: _mul_dual,
    np.divide: _div_dual,
    np.power: _pow_dual,
}

DUAL_COMPARE_OPS = {
    np.equal: _eq_dual,
    np.not_equal: _neq_dual,
    np.less: _lt_dual,
    np.less_equal: _leq_dual,
    np.greater: _gt_dual,
    np.greater_equal: _geq_dual,
}

DUAL_UNARY_OPS = {
    np.sin: _sin_dual,
    np.cos: _cos_dual,
    np.exp: _exp_dual,
    np.log: _log_dual,
    np.sqrt: _sqrt_dual,
}

class Dual(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, a, b):
        self.a = np.asarray(a)
        self.b = np.asarray(b)

        if self.a.ndim > 0 and self.b.ndim == 0:
            self.b = np.full(self.a.shape, b)

        if self.a.shape != self.b.shape:
            raise ValueError('a and b must have the same shape')

    @property
    def shape(self):
        return self.a.shape

    @property
    def ndim(self):
        return self.a.ndim

    def __str__(self):
        if self.ndim == 0:
            return f'{self.a}, {self.b}'

        if self.ndim == 1:
            s = '[ '

            threshold = np.get_printoptions()['threshold']
            if self.a.size < threshold:
                for a, b in zip(self.a[:threshold], self.b[:threshold]):
                    s += f'{a}, {b}\n  '
                s = s[:-3]
            else:
                s += f'{self.a[0]}, {self.b[0]}\n  '
                s += f'{self.a[1]}, {self.b[1]}\n  '
                s += '...\n  '
                s += f'{self.a[-2]}, {self.b[-2]}\n  '
                s += f'{self.a[-1]}, {self.b[-1]}'

            s += ' ]'

            return s

        return f'Dual object with shape {self.shape}'

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented

        if ufunc in DUAL_BINARY_OPS:
            x = asdual(inputs[0])
            y = asdual(inputs[1])

            return DUAL_BINARY_OPS[ufunc](x, y)

        elif ufunc in DUAL_COMPARE_OPS:
            x = asdual(inputs[0])
            y = asdual(inputs[1])

            return DUAL_COMPARE_OPS[ufunc](x, y)

        elif ufunc in DUAL_UNARY_OPS:
            return DUAL_UNARY_OPS[ufunc](self)

        else:
            return NotImplemented
