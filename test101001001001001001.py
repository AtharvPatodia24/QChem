import qiskit_nature
from qiskit_nature.second_q.drivers import Psi4Driver
import numpy as np
from qiskit_nature.units import DistanceUnit
from qiskit_algorithms import VQE
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from matplotlib import pyplot as plt
import psi4
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import SLSQP
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature import settings
settings.use_ipython = False
energies = {}
service = QiskitRuntimeService(channel="ibm_quantum", token="82f6eeb5a67e68bc27f19c170479d281103dddfc005292906679d90a8971f33b84ce8c51ce991c90181edd9734ecc5e9519afc001f203c9447e8247923124966")
backend = service.least_busy(operational=True, simulator=False)
molecule_geometry = f"""
0 1
Li 0.0 0.0 0.5
H  0.0 0.0 0.75
    """


molecule_geometry = molecule_geometry.strip()



print(molecule_geometry)


psi4_input = f"""
molecule {{
{molecule_geometry}
}}

set basis = sto-3g

energy('scf', return_wfn=True)
    """

driver = Psi4Driver(psi4_input)
es_problem = driver.run()
print(es_problem)
mapper = JordanWignerMapper()
ansatz = UCCSD(
    es_problem.num_spatial_orbitals,
    es_problem.num_particles,
    mapper,
    initial_state=HartreeFock(
        es_problem.num_spatial_orbitals,
        es_problem.num_particles,
        mapper,
    ),
)

vqe_solver = VQE(Estimator( ), ansatz, SLSQP())
vqe_solver.initial_point = [0.0] * ansatz.num_parameters
calc = GroundStateEigensolver(mapper, vqe_solver)
res = calc.solve(es_problem)
print(res)

