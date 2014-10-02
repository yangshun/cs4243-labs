import numpy as np
from math import *

def deg_to_rad(deg):
  # Converts an angle from degrees to radians
  return float(deg)/180 * pi

def conjugate(quat):
  # Calculates the conjugate of a quaternion
  return quat[0:1] + np.negative(quat[1:]).tolist()

def approx(value):
  # Approximate a floating point value
  val = round(value, 6)
  return 0.0 if val == 0.0 else val

def approx_mat(mat):
  # Approximate every value in a matrix
  for pt in np.nditer(mat, op_flags=['readwrite']):
    pt[...] = approx(pt)
  return mat

def quatmult(p, q):
  # Performs the multiplication of two quaternions
  s_p, v_p = p[0], np.array(p[1:])
  s_q, v_q = q[0], np.array(q[1:])
  s_pq = s_q * s_p - np.dot(v_q, v_p)
  v_pq = np.cross(v_p, v_q) + s_q * v_p + s_p * v_q
  out = [s_pq]
  out.extend(v_pq)
  return out

def quatmult_2(p, q):
  # Alternative version of quaternion multiplication
  out = [0] * 4
  out[0] = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
  out[1] = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
  out[2] = p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1]
  out[3] = p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]
  return out

def quatrot(p, q):
  # Performs rotation of two quaternions
  # p is the point to be rotated and q is the rotation quaternion
  return [approx(x) for x in quatmult(quatmult(q, p), conjugate(q))]

def quat2rot(q):
  # Returns a 3x3 rotation matrix parameterized with
  # the elements of an input quaternion
  q0, q1, q2, q3 = q
  return np.matrix([[q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
                    [2*(q1*q2 + q0*q3), q0**2 + q2**2 - q1**2 - q3**2, 2*(q2*q3 - q0*q1)],
                    [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 + q3**2 - q1**2 - q2**2]])

pts = np.zeros([11, 3]) 
pts[0, :] = [-1, -1, -1]
pts[1, :] = [1, -1, -1]
pts[2, :] = [1, 1, -1]
pts[3, :] = [-1, 1, -1]
pts[4, :] = [-1, -1, 1]
pts[5, :] = [1, -1, 1]
pts[6, :] = [1, 1, 1]
pts[7, :] = [-1, 1, 1]
pts[8, :] = [-0.5, -0.5, -1] 
pts[9, :] = [0.5, -0.5, -1] 
pts[10, :] = [0, 0.5, -1]

# Calculating camera positions for each frame
initial_pos = [0, 0, 0, -5]
camera_pos = [initial_pos]
camera_rot_quat = [cos(deg_to_rad(-15)), 0, sin(deg_to_rad(-15)), 0]
pos_new = initial_pos

for i in range(3):
  pos_new = quatrot(pos_new, camera_rot_quat)
  camera_pos.append(pos_new)

# Camera positions for each frame
pos_1, pos_2, pos_3, pos_4 = camera_pos

# Calculating camera orientation for each frame
initial_orntn = np.identity(3)
camera_orntns = [initial_orntn]
camera_rot_mat = quat2rot([cos(deg_to_rad(15)), 0, sin(deg_to_rad(15)), 0])
orntn_new = initial_orntn

for i in range(3):
  orntn_new = camera_rot_mat * orntn_new
  camera_orntns.append(orntn_new)

camera_orntns = [approx_mat(m) for m in camera_orntns]
# Camera orientations for each frame
quatmat_1, quatmat_2, quatmat_3, quatmat_4 = camera_orntns
