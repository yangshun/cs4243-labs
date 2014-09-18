# CS4243 Lab 1
# ===================
# Name: Tay Yang Shun
# Matric: A0073063M

import numpy as np
import numpy.linalg as la

file = open('data.txt')
data = np.genfromtxt(file, delimiter=',')
file.close()

print "Data =\n", data

def rowify(m):
  half = np.append(np.array(m), 1)
  return np.append(np.append(half, np.zeros(6)), half)

M = np.apply_along_axis(rowify, 1, np.array(data[:,:2])).flatten().reshape(-1, 6)
b = np.matrix(data[:,2:]).flatten().transpose()

print "M =\n", M
print "b =\n", b

a, e, r, s = la.lstsq(M, b)

print "Least-square solution, a =\n", a
print
print "Residue =\n", e
print
print "Rank =\n", r
print
print "Singular values =\n", s
print
print "Sum-squared error:", la.norm(M * a - b) ** 2
