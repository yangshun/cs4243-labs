import numpy as np

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

def quatmult(q, p):
  s_q, v_q = q[0], np.array(q[1:])
  s_p, v_p = p[0], np.array(p[1:])
  s_qp = s_q * s_p - np.dot(v_q, v_p)
  v_qp = np.cross(v_q, v_p) + s_q * v_p + s_p * v_q
  out = [s_qp]
  out.extend(v_qp)
  return out

def quatmult2(p, q):
  # Version 2
  out = [0] * 4
  out[0] = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
  out[1] = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
  out[2] = p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1]
  out[3] = p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]
  return out
