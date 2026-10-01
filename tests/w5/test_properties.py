"""Week 5 -- YOUR property-based tests for forge.autograd (hypothesis).

Write at least 5 @given properties. One example is here; ideas for the rest:
  * linearity: grad of (a*f(x) + b*g(x)) == a*grad f + b*grad g
  * d/dx sum(x * c) == c for any c (broadcast shapes too -- generate shapes with hypothesis.extra.numpy)
  * unbroadcast(g, shape).sum() == g.sum() for any broadcastable pair
  * log_softmax(x + c) == log_softmax(x) for any constant c (shift invariance)
  * exp(log_softmax(x)).sum(axis) == 1
  * backward twice == 2x the gradient of backward once (accumulation)
"""
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from forge.autograd import Tensor

finite = st.floats(-10, 10, allow_nan=False, allow_infinity=False)


@given(arrays(np.float64, st.integers(1, 20), elements=finite))
def test_grad_of_sum_of_squares_is_2x(x):
    t = Tensor(x.copy(), requires_grad=True)
    (t * t).sum().backward()
    np.testing.assert_allclose(t.grad, 2 * x)
