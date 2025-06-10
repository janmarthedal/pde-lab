from scipy.integrate import dblquad

def f(x, y):
    return 2 * x + y

v = dblquad(f, 1, 3, lambda y: (y + 1) / 2, lambda y: (7 - y) / 2)
print(v)

w = dblquad(lambda a, b: 4 * f(1 + a + 2 * b, 1 + 2 * a), 0, 1, 0, lambda a: 1 - a)
print(w)
