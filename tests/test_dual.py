import numpy as np
import pytest
from dual import Dual

def check(z, a, b=None):
    assert np.allclose(z.a, a)
    if b is not None:
        assert np.allclose(z.b, b)

BINARY_CASES = [
    (np.add, Dual(1, 0.1), Dual(2, 0.2), 3, 0.3),
    (np.add, Dual(1, 0.1), 2, 3, 0.1),
    (np.add, 2, Dual(1, 0.1), 3, 0.1),

    (np.subtract, Dual(2, 0.2), Dual(1, 0.1), 1, 0.1),
    (np.subtract, Dual(2, 0.2), 1, 1, 0.2),

    (np.multiply, Dual(2, 0.2), Dual(3, 0.3), 6, 1.2),
    (np.multiply, Dual(2, 0.2), 3, 6, 0.6),

    (np.divide, Dual(4, 0.4), Dual(2, 0.2), 2, 0.0),
    (np.divide, Dual(4, 0.4), 2, 2, 0.2),
]

@pytest.mark.parametrize('ufunc,x,y,a,b', BINARY_CASES)
def test_binary(ufunc, x, y, a, b):
    check(ufunc(x, y), a, b)

def test_power_const_exp():
    x = Dual(3, 0.3)
    z = np.power(x, 2)

    check(z, 9, 1.8)

def test_power_dual_exp():
    x = Dual(3, 0.3)
    y = Dual(2, 0.2)

    z = np.power(x, y)

    check(z, 9, 1.8 * (1 + np.log(3)))

UNARY_CASES = [
    (np.sin, Dual(0.5, 1.0), np.sin(0.5), np.cos(0.5)),
    (np.cos, Dual(0.5, 1.0), np.cos(0.5), -np.sin(0.5)),
    (np.exp, Dual(2.0, 1.0), np.exp(2.0), np.exp(2.0)),
    (np.log, Dual(2.0, 1.0), np.log(2.0), 1 / 2.0),
    (np.sqrt, Dual(4.0, 1.0), 2.0, 1 / (2 * np.sqrt(4.0))),
]

@pytest.mark.parametrize('ufunc,x,a,b', UNARY_CASES)
def test_unary(ufunc, x, a, b):
    check(ufunc(x), a, b)

COMPARE_CASES = [
    (np.less, Dual(1, 0.1), Dual(2, 0.2), True),
    (np.less, Dual(2, 0.2), Dual(1, 0.1), False),
    (np.less_equal, Dual(2, 0.2), Dual(2, 0.2), True),
    (np.less_equal, Dual(2, 0.2), Dual(1, 0.1), False),
    (np.greater, Dual(3, 0.3), Dual(2, 0.2), True),
    (np.greater, Dual(2, 0.2), Dual(3, 0.3), False),
    (np.greater_equal, Dual(2, 0.2), Dual(2, 0.2), True),
    (np.greater_equal, Dual(2, 0.2), Dual(3, 0.3), False),

    (np.equal, Dual(2, 0.2), Dual(2, 0.3), True),
    (np.not_equal, Dual(2, 0.2), Dual(3, 0.2), True),
    (np.not_equal, Dual(2, 0.2), Dual(2, 0.3), False),
    (np.equal, Dual(2, 0.2), Dual(3, 0.2), False),
]

@pytest.mark.parametrize('ufunc,x,y,expected', COMPARE_CASES)
def test_comparisons(ufunc, x, y, expected):
    assert np.array_equal(ufunc(x, y), expected)

def test_broadcast():
    x = Dual(np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3]))
    y = 2.0

    z = x + y

    assert z.a.shape == (3,)
    assert z.b.shape == (3,)

    check(z, [3, 4, 5], [0.1, 0.2, 0.3])

def test_symmetry():
    x = Dual(1, 0.1)
    y = Dual(2, 0.2)

    z1 = x + y
    z2 = y + x

    check(z1, z2.a, z2.b)
