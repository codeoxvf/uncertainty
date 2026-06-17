import numpy as np
from dual import Dual, DUAL_BINARY_OPS, DUAL_UNARY_OPS

def uncertain(x):
    if isinstance(x, Uncertain):
        return x
    return Uncertain(x, 0.0)

class Uncertain(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, mean, sd, correlations=None):
        self.mean = np.asarray(mean)
        self.sd = np.asarray(sd)

        if self.mean.shape != self.sd.shape:
            raise ValueError('mean and sd must have the same shape')

        if correlations is None:
            self.correlations = { self: 1 }
        else:
            self.correlations = correlations
            self.correlations[self] = 1

    @property
    def relerr(self):
        return self.sd / self.mean

    @property
    def shape(self):
        return self.mean.shape

    @property
    def ndim(self):
        return self.mean.ndim
    
    def add_corr(self, other, corr):
        if not isinstance(other, Uncertain):
            raise TypeError('Uncertains can only be correlated to other Uncertains')
        self.correlations[other] = corr
    
    def __hash__(self):
        return id(self)

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

        if ufunc in DUAL_BINARY_OPS:
            x, y = inputs
            x = uncertain(x)
            y = uncertain(y)

            # calculate partial derivatives with autodiff
            dxf = ufunc(Dual(x.mean, 1), Dual(y.mean, 0)).b
            dyf = ufunc(Dual(x.mean, 0), Dual(y.mean, 1)).b

            var = (dxf * x.sd)**2 + (dyf * y.sd)**2
            if y in x.correlations:
                var += 2 * x.correlations[y] * x.sd * y.sd * dxf * dyf

            mean = ufunc(x.mean, y.mean)
            sd = np.sqrt(var)

            return Uncertain(mean, sd)

        elif ufunc in DUAL_UNARY_OPS:
            mean = ufunc(self.mean)
            sd = np.abs(ufunc(Dual(self.mean, 1)).b) * self.sd

            return Uncertain(mean, sd)

        else:
            return NotImplemented
