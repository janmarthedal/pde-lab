from scipy.linalg import solve, inv
from scipy.integrate import dblquad

# Return a, b, c so that
#  f(p1) = 1, f(p2) = 0, f(p3) = 0
# for
#  f(x,y) = a*x + b*y + c
def plane_coef(p1, p2, p3):
    return solve([[p[0], p[1], 1] for p in [p1, p2, p3]], [1, 0, 0])

def line_coef(p1, p2):
    return solve([[p1[1], 1], [p2[1], 1]], [p1[0], p2[0]])

def compute_args(points: list[tuple[int, int]]):
    points.sort(key=lambda p: p[1])
    return [[p[1] for p in points], line_coef(points[0], points[1]), line_coef(points[1], points[2]), line_coef(points[0], points[2])]

def x_limit(y, left_limit: bool, args):
    if y < args[0][1]:
        x1 = args[1][0] * y + args[1][1]
    else:
        x1 = args[2][0] * y + args[2][1]
    x2 = args[3][0] * y + args[3][1]
    return min(x1, x2) if left_limit else max(x1, x2)

def eval_plane(x, y, plane):
    return plane[0] * x + plane[1] * y + plane[2]

p1 = (1, 2)
p2 = (5, 1)
p3 = (3, 5)
plane1 = plane_coef(p1, p2, p3)
plane2 = plane_coef(p2, p3, p1)
plane3 = plane_coef(p3, p1, p2)
args = compute_args([p1, p2, p3])

# v = dblquad(lambda x, y: eval_plane(x, y, plane1) * eval_plane(x, y, plane1), args[0][0], args[0][2], lambda y: x_limit(y, True, args), lambda y: x_limit(y, False, args))
# v = dblquad(lambda x, y: plane1[0] * plane1[0] + plane1[1] * plane1[1], args[0][0], args[0][2], lambda y: x_limit(y, True, args), lambda y: x_limit(y, False, args))
v = dblquad(lambda x, y: plane1[0] * plane2[0] + plane1[1] * plane2[1], args[0][0], args[0][2], lambda y: x_limit(y, True, args), lambda y: x_limit(y, False, args))
print(v)

dx1 = p2[0] - p1[0]
dy1 = p2[1] - p1[1]
dx2 = p3[0] - p1[0]
dy2 = p3[1] - p1[1]
z1 = solve([[dx1, dy1], [dx2, dy2]], [-1, -1])
z2 = solve([[dx1, dy1], [dx2, dy2]], [1, 0])
j = z1[0] * z2[0] + z1[1] * z2[1]
c = abs(dx1 * dy2 - dx2 * dy1)
w = dblquad(lambda s, t: j, 0, 1, 0, lambda s: 1 - s)
print(w)
print(w[0] * c)
print(c * j / 2)
