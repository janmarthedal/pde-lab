from numpy import array
from .base_element import BaseElement


class Quadrilateral(BaseElement):

    def eval(self, p):
        s, t = p
        return 0.25 * array(
            [
                (1.0 - s) * (1.0 - t),
                (1.0 + s) * (1.0 - t),
                (1.0 + s) * (1.0 + t),
                (1.0 - s) * (1.0 + t),
            ]
        )

    def grad(self, p):
        s, t = p
        return 0.25 * array(
            [
                [-1.0 + t, -1.0 + s],
                [ 1.0 - t, -1.0 - s],
                [ 1.0 + t,  1.0 + s],
                [-1.0 - t,  1.0 - s],
            ]
        )
