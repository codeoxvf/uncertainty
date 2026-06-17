# Uncertainty propagation in Python

Propagation of uncertainty through numerical calculations using first-order Taylor approximation: for a function $f(x,y)$,

```math
\sigma_f^2 = \left( \frac{\partial f}{\partial x} \right)^2 \sigma_x^2 + \left( \frac{\partial f}{\partial y} \right)^2 \sigma_y^2 + 2 \frac{\partial f}{\partial x} \frac{\partial f}{\partial y} \sigma_{xy}.
```

Calculates partial derivatives using forward-mode automatic differentiation, implemented from scratch (class `Dual`).

## Example

```python-repl
>>> from uncertainty import Uncertain

>>> x = Uncertain(1, 0.5)
>>> y = Uncertain(1, 0.5)

>>> x - y
0 ± 0.7071067811865476

>>> x.add_corr(y, 1)
>>> x - y
0 ± 0.0
```
