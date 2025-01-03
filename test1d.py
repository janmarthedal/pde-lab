from typing import Iterable
import numpy as np
from scipy import integrate, sparse
from scipy.sparse.linalg import spsolve

def lagrange_1_00(x: float) -> float:
    return (1 - x) ** 2 / 4

def lagrange_1_01(x: float) -> float:
    return (1 - x) * (1 + x) / 4

def dlagrange_1_00(x: float) -> float:
    return 1/4

def dlagrange_1_01(x: float) -> float:
    return -1/4

def make_1d_mesh(a: float, b: float, element_count: int) -> tuple[list[float], list[tuple[int, int]]]:
    points = [a + i * (b - a) / element_count for i in range(element_count)]
    points.append(b)
    elements = list(zip(range(element_count), range(1, element_count + 1)))
    return points, elements

def integrate_prod(local1, local2, element, points):
    a = points[element[local1]]
    b = points[element[1 if local1 == 0 else 0]]
    dx = abs(b - a) / 2
    if local1 != local2:
        v = integrate.quad(lagrange_1_01, -1, 1)
    else:
        v = integrate.quad(lagrange_1_00, -1, 1)
    return v[0] * dx

def integrate_div(local1, local2, element, points) -> float:
    a = points[element[local1]]
    b = points[element[1 if local1 == 0 else 0]]
    dx = abs(b - a) / 2
    if local1 != local2:
        v = integrate.quad(dlagrange_1_01, -1, 1)
    else:
        v = integrate.quad(dlagrange_1_00, -1, 1)
    return v[0] / dx

def local_element_entries(element) -> int:
    elem_size = len(element)
    return elem_size * (elem_size + 1) // 2

def integrate_line(element, points, value_type) -> Iterable[tuple[int, int, float]]:
    elem_size = len(element)
    for i in range(elem_size):
        gi = element[i]
        for j in range(i, elem_size):
            gj = element[j]
            if value_type == 'div':
                v = integrate_div(i, j, element, points)
            elif value_type == 'prod':
                v = integrate_prod(i, j, element, points)
            else:
                assert False
            yield gi, gj, v

def bilinear_to_matrix(elements, points, value_type: str, multiplier=1.0):
    total_element_entries = sum(local_element_entries(element) for element in elements)
    entries = np.empty(total_element_entries, dtype=[('row',np.int32),('col',np.int32),('val',np.float64)])
    k = 0
    for element in elements:
        for gi, gj, v in integrate_line(element, points, value_type):
            if gi > gj:
                gi, gj = gj, gi
            entries[k] = gi, gj, v * multiplier
            k += 1
    assert k == total_element_entries
    entries.sort(order=['row', 'col'])
    S = sparse.csr_matrix((entries['val'], (entries['row'], entries['col'])), shape=(len(points), len(points)))
    S += sparse.triu(S, 1).T
    return S

# `indices` must be sorted in ascending order
def enforce_rows(C, indices):
    if not indices:
        return C
    blocks = []
    i = 0
    for row in indices:
        if i < row:
            blocks.append(C[i:row])
        blocks.append(sparse.eye(1, C.shape[1], row))
        i = row + 1
    if i < C.shape[0]:
        blocks.append(C[i:])
    return sparse.vstack(blocks, format='csr')

points, elements = make_1d_mesh(0, 1, 5)
print("points", points)
print("elements", elements)
print("Total element entries", sum(local_element_entries(element) for element in elements))

K = bilinear_to_matrix(elements, points, "div")
M = bilinear_to_matrix(elements, points, "prod", 4)
C = K + M

print(C.toarray())

R = np.zeros((len(points),))
R[0] = 1
R[5] = 2
C = enforce_rows(C, [0, 5])

print(C.toarray())
print(R)

a = spsolve(C, R)

print(a)
