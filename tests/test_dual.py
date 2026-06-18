import numpy as np
import pytest
from dual import Dual, asdual

def assert_dual_eq(x, a, b=None):
    assert isinstance(x, Dual)

    np.testing.assert_allclose(x.a, a, atol=1e-12)

    if b is not None:
        np.testing.assert_allclose(x.b, b, atol=1e-12)

# ==================
# structural tests

@pytest.mark.parametrize('ufunc,inputs,expected_shape', [
    pytest.param(np.add,
        (Dual(1, 0.1), 2),
        (),
        id='uscalar-scalar'),
    pytest.param(np.subtract,
        (Dual(np.array([1, 2, 3]), 0.1), 2),
        (3,),
        id='uvector-scalar'),
    pytest.param(np.multiply,
        (Dual(2, 0.2), np.array([1, 2, 3])),
        (3,),
        id='uscalar-vector'),
    pytest.param(np.divide,
        (Dual(np.array([1, 2, 3]), 0.1),
           np.array([1, 2, 3])),
        (3,),
        id='uvector-vector'),
    pytest.param(np.power,
        (Dual(np.array([1, 2, 3]), 0.1),
            Dual(np.array([[1], [2]]), np.array([[0.1], [0.2]]))),
        (2,3),
        id='uvector-broadcast'),
    pytest.param(np.exp,
        (Dual(np.array([1, 2, 3]), 0.1),),
        (3,),
        id='unary-preserves'),
])
def test_type_shape(ufunc, inputs, expected_shape):
    x = ufunc(*inputs)

    assert isinstance(x, Dual)
    assert x.shape == expected_shape

    assert np.all(x.b >= 0)

def test_asdual():
    x = asdual(5)
    y = asdual(Dual(1, 0.1))

    assert_dual_eq(x, 5, 0)
    assert_dual_eq(y, 1, 0.1)

@pytest.mark.parametrize('a,b', [
    pytest.param(1, [1, 2], id='scalar-vector'),
    pytest.param([1, 2, 3], [0.1, 0.2], id='shape-mismatch'),
])
def test_init(a, b):
    with pytest.raises(ValueError):
        x = Dual(a, b)

def test_const_b():
    x = Dual([1, 2, 3], 0.1)
    assert x.shape == (3,)

def test_subscript():
    x = Dual([1, 2, 3], 0.1)

    assert_dual_eq(x[1], 2, 0.1)
    assert x[1].shape == ()

    assert isinstance(x[1:], Dual)
    assert x[1:].shape == (2,)

# ==================
# mathematical tests

def test_degenerate():
    x = Dual(2, 0.0)
    y = Dual(3, 0.0)

    z = x + y

    assert_dual_eq(z, 5, 0.0)

def test_binary():
    x = Dual(2, 0.1)
    y = Dual(3, 0.2)

    z1 = x + y
    z2 = x * y

    assert_dual_eq(z1, 5, 0.3)
    assert_dual_eq(z2, 6, 0.7)
    assert x != y

def test_unary():
    x = Dual(np.pi, 0.1)

    y = np.sin(x)

    assert_dual_eq(y, 0.0, -0.1)

def test_broadcast():
    x = Dual(np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3]))
    y = Dual(3, 0.2)

    z = x * y

    assert_dual_eq(z, np.array([3, 6, 9]), np.array([0.5, 1.0, 1.5]))

def test_broadcast_const_b():
    x = Dual(np.array([1, 2, 3]), np.array(0.1))
    y = Dual(3, 0.2)

    z = x * y

    assert_dual_eq(z, np.array([3, 6, 9]), np.array([0.5, 0.7, 0.9]))

def test_symmetry():
    x = Dual(2, 0.1)
    y = Dual(3, 0.2)

    z1 = x + y
    z2 = y + x

    assert_dual_eq(z1, z2.a, z2.b)

def test_composition():
    x = Dual(2, 0.1)
    y = np.exp(x**2)

    assert_dual_eq(y, np.exp(4), 0.4*np.exp(4))

@pytest.mark.parametrize('ufunc,a,b', [
    pytest.param(np.sqrt, 0, np.inf, id='sqrt'),
    pytest.param(lambda x: 1/x, np.inf, -np.inf, id='1/uscalar-0div'),
    pytest.param(lambda x: 0/x, np.nan, np.nan, id='scalar/uscalar'),
    pytest.param(lambda x: x/0, np.nan, np.nan, id='uscalar/scalar'),
    pytest.param(np.log, -np.inf, np.inf, id='log'),
])
def test_zero(ufunc, a, b):
    with pytest.warns(RuntimeWarning,
            match=r'(divide by zero|invalid value) encountered'):
        assert_dual_eq(ufunc(Dual(0, 0.1)), a, b)

def test_other_zero():
    x = Dual(0, 0.1)
    y = Dual(1, 0.1)

    assert_dual_eq(x**2, 0, 0.0)

    with pytest.warns(RuntimeWarning,
            match=r'(divide by zero|invalid value) encountered'):
        # assert_dual_eq(y/0, np.inf, np.inf)
        assert_dual_eq(y/0, np.inf, np.nan)

@pytest.mark.parametrize('ufunc,a,b', [
    pytest.param(np.log, np.nan, -0.1, id='log'),
    pytest.param(np.sqrt, np.nan, np.nan, id='sqrt'),
    pytest.param(lambda x: x**0.5, np.nan, np.nan, id='fractional-power'),
])
def test_imag(ufunc, a, b):
    with pytest.warns(RuntimeWarning,
            match=r'(divide by zero|invalid value) encountered'):
        assert_dual_eq(ufunc(Dual(-1, 0.1)), a, b)
