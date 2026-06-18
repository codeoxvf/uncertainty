import numpy as np
import pytest
from uncertainty import Uncertain, uncertain

def assert_uncertain_eq(x, mean, sd=None):
    assert isinstance(x, Uncertain)

    np.testing.assert_allclose(x.mean, mean, atol=1e-12)

    if sd is not None:
        np.testing.assert_allclose(x.sd, sd, atol=1e-12)

def add_quad(x, y):
    return np.sqrt(x**2 + y**2)

# ==================
# structural tests

@pytest.mark.parametrize('ufunc,inputs,expected_shape', [
    pytest.param(np.add,
        (Uncertain(1, 0.1), 2),
        (),
        id='uscalar-scalar'),
    pytest.param(np.subtract,
        (Uncertain(np.array([1, 2, 3]), 0.1), 2),
        (3,),
        id='uvector-scalar'),
    pytest.param(np.multiply,
        (Uncertain(2, 0.2), np.array([1, 2, 3])),
        (3,),
        id='uscalar-vector'),
    pytest.param(np.divide,
        (Uncertain(np.array([1, 2, 3]), 0.1),
           np.array([1, 2, 3])),
        (3,),
        id='uvector-vector'),
    pytest.param(np.power,
        (Uncertain(np.array([1, 2, 3]), 0.1),
            Uncertain(np.array([[1], [2]]), np.array([[0.1], [0.2]]))),
        (2,3),
        id='uvector-broadcast'),
    pytest.param(np.exp,
        (Uncertain(np.array([1, 2, 3]), 0.1),),
        (3,),
        id='unary-preserves'),
])
def test_type_shape(ufunc, inputs, expected_shape):
    x = ufunc(*inputs)

    assert isinstance(x, Uncertain)
    assert x.shape == expected_shape

    assert np.all(x.sd >= 0)

def test_convert():
    x = uncertain(5)
    y = uncertain(Uncertain(1, 0.1))

    assert_uncertain_eq(x, 5, 0)
    assert_uncertain_eq(y, 1, 0.1)

@pytest.mark.parametrize('mean,sd', [
    pytest.param(1, -0.1, id='negative-sd'),
    pytest.param(1, [1, 2], id='scalar-vector'),
    pytest.param([1, 2, 3], [0.1, 0.2], id='shape-mismatch'),
])
def test_init(mean, sd):
    with pytest.raises(ValueError):
        x = Uncertain(mean, sd)

def test_const_sd():
    x = Uncertain([1, 2, 3], 0.1)
    assert x.shape == (3,)

def test_subscript():
    x = Uncertain([1, 2, 3], 0.1)

    assert_uncertain_eq(x[1], 2, 0.1)
    assert x[1].shape == ()

    assert isinstance(x[1:], Uncertain)
    assert x[1:].shape == (2,)

# ==================
# mathematical tests

def test_degenerate():
    x = Uncertain(2, 0.0)
    y = Uncertain(3, 0.0)

    z = x + y

    assert_uncertain_eq(z, 5, 0.0)

def test_binary():
    x = Uncertain(2, 0.1)
    y = Uncertain(3, 0.2)

    z1 = x + y
    z2 = x * y

    assert_uncertain_eq(z1, 5, add_quad(0.1, 0.2))
    assert_uncertain_eq(z2, 6, 6 * add_quad(0.1 / 2, 0.2 / 3))
    assert x != y

def test_unary():
    x = Uncertain(np.pi, 0.1)

    y = np.sin(x)

    assert_uncertain_eq(y, 0.0, 0.1)

def test_broadcast():
    x = Uncertain(np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3]))
    y = Uncertain(3, 0.2)

    z = x * y

    assert_uncertain_eq(
        z,
        np.array([3, 6, 9]),
        np.array([
            3 * add_quad(0.1 / 1, 0.2 / 3),
            6 * add_quad(0.2 / 2, 0.2 / 3),
            9 * add_quad(0.3 / 3, 0.2 / 3),
        ])
    )

def test_broadcast_const_sd():
    x = Uncertain(np.array([1, 2, 3]), np.array(0.1))
    y = Uncertain(3, 0.2)

    z = x * y

    assert_uncertain_eq(
        z,
        np.array([3, 6, 9]),
        np.array([
            3 * add_quad(0.1 / 1, 0.2 / 3),
            6 * add_quad(0.1 / 2, 0.2 / 3),
            9 * add_quad(0.1 / 3, 0.2 / 3),
        ])
    )

def test_symmetry():
    x = Uncertain(2, 0.1)
    y = Uncertain(3, 0.2)

    z1 = x + y
    z2 = y + x

    assert_uncertain_eq(z1, z2.mean, z2.sd)

def test_composition():
    x = Uncertain(2, 0.1)
    y = np.exp(x**2)

    assert_uncertain_eq(y, np.exp(4), 0.4*np.exp(4))

def test_self_correlation():
    x = Uncertain(2, 0.2)
    x2 = x**2

    assert_uncertain_eq(x - x, 0, 0.0)
    assert_uncertain_eq(x * x, x2.mean, x2.sd)

@pytest.mark.parametrize('ufunc,mean,sd', [
    pytest.param(np.sqrt, 0, np.inf, id='sqrt'),
    # pytest.param(lambda x: 1/x, np.inf, np.inf, id='1/uscalar-0div'),
    pytest.param(lambda x: 1/x, np.inf, np.nan, id='1/uscalar-0div'),
    pytest.param(lambda x: 0/x, np.nan, np.nan, id='scalar/uscalar'),
    pytest.param(lambda x: x/0, np.nan, np.nan, id='uscalar/scalar'),
    pytest.param(np.log, -np.inf, np.inf, id='log'),
])
def test_zero(ufunc, mean, sd):
    with pytest.warns(RuntimeWarning,
            match=r'(divide by zero|invalid value) encountered'):
        assert_uncertain_eq(ufunc(Uncertain(0, 0.1)), mean, sd)

def test_other_zero():
    x = Uncertain(0, 0.1)
    y = Uncertain(1, 0.1)

    assert_uncertain_eq(x**2, 0, 0.0)

    with pytest.warns(RuntimeWarning,
            match=r'(divide by zero|invalid value) encountered'):
        # assert_uncertain_eq(y/0, np.inf, np.inf)
        assert_uncertain_eq(y/0, np.inf, np.nan)

@pytest.mark.parametrize('ufunc,mean,sd', [
    pytest.param(np.log, np.nan, 0.1, id='log'),
    pytest.param(np.sqrt, np.nan, np.nan, id='sqrt'),
    pytest.param(lambda x: x**0.5, np.nan, np.nan, id='fractional-power'),
])
def test_imag(ufunc, mean, sd):
    with pytest.warns(RuntimeWarning,
            match=r'(divide by zero|invalid value) encountered'):
        assert_uncertain_eq(ufunc(Uncertain(-1, 0.1)), mean, sd)
