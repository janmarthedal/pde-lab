from numpy import array
from .base_integrator import BaseIntegrator

class Triangle0Integrator(BaseIntegrator):

    # exact for constant-valued `fun`
    def integrate(self, fun: callable) -> float:
        return 0.5 * fun(array([0., 0.]))
