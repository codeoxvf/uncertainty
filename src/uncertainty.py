import numpy as np
from dual import Dual

def uncertain(x):
    if isinstance(x, Uncertain):
        return x
    return Uncertain(x, 0.0)

UNCERTAIN_BINARY_OPS = set([
    np.add,
    np.subtract,
    np.multiply,
    np.divide,
    np.power,
    np.equal,
    np.not_equal,
    np.less,
    np.less_equal,
    np.greater,
    np.greater_equal,
])

UNCERTAIN_UNARY_OPS = set([
    np.sin,
    np.cos,
    np.exp,
    np.log,
    np.sqrt,
])

class Uncertain(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, mean, sd, correlations=None):
        self.mean = np.asarray(mean)
        self.sd = np.asarray(sd)

        if correlations is None:
            self.correlations = set()
        else:
            self.correlations = correlations

    @property
    def relerr(self):
        return self.sd / self.mean

    @property
    def shape(self):
        return self.mean.shape

    @property
    def ndim(self):
        return self.mean.ndim

    def __repr__(self):
        return f'Uncertain(mean={self.mean}, sd={self.sd})'
    
    def __str__(self):
        if self.ndim == 0:
            return f'{self.mean} ± {self.sd}'

        if self.ndim == 1:
            s = '[ '

            threshold = np.get_printoptions()['threshold']
            if self.a.size < threshold:
                for a, b in zip(self.a[:threshold], self.b[:threshold]):
                    s += f'{a}, {b}\n  '
                s = s[:-3]
            else:
                s += f'{self.a[0]} ± {self.b[0]}\n  '
                s += f'{self.a[1]} ± {self.b[1]}\n  '
                s += '...\n  '
                s += f'{self.a[-2]} ± {self.b[-2]}\n  '
                s += f'{self.a[-1]} ± {self.b[-1]}'

            s += ' ]'

            return s

        return f'Uncertain object with shape {self.shape}'

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented

        if ufunc in UNCERTAIN_BINARY_OPS:
            x, y = inputs
            x = uncertain(x)
            y = uncertain(y)

            # calculate partial derivatives with autodiff
            dxf = ufunc(Dual(x.mean, 1), Dual(y.mean, 0)).b
            dyf = ufunc(Dual(x.mean, 0), Dual(y.mean, 1)).b

            mean = ufunc(x.mean, y.mean)
            sd = np.sqrt((dxf * x.sd)**2 + (dyf * y.sd)**2)

            return Uncertain(mean, sd)
        elif ufunc in UNCERTAIN_UNARY_OPS:
            mean = ufunc(self.mean)
            sd = np.abs(ufunc(Dual(self.mean, 1)).b) * self.sd

            return Uncertain(mean, sd)
        else:
            return NotImplemented
