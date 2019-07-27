# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 08:21:58 2019

@author: elif.ayvali
"""

import numpy as np
import math
from statsmodels.stats.correlation_tools import cov_nearest


class Tools:
    
    def nearestPSD(P):
        #other options:not ideal but necessary for robust solutions:
        #1) P?1/2P+1/2P' to even out the off-diagonal terms -- for symmetry
        #2)Let P=P+eps In×n, where eps is a small scalar to make sure matrix is not ill conditioned
        #3) use 64fp arithmetic
        return cov_nearest(P)
    
    def rot_to_quat(rot, isprecise=False):
        """Return quaternion from rotation matrix.
        If isprecise is True, the input matrix is assumed to be a precise rotation
        matrix and a faster algorithm is used.
        """
        M = np.array(rot, dtype=np.float64, copy=False)[:4, :4]
        if isprecise:
            q = np.empty((4, ))
            t = np.trace(M)
            if t > M[3, 3]:
                q[0] = t
                q[3] = M[1, 0] - M[0, 1]
                q[2] = M[0, 2] - M[2, 0]
                q[1] = M[2, 1] - M[1, 2]
            else:
                i, j, k = 0, 1, 2
                if M[1, 1] > M[0, 0]:
                    i, j, k = 1, 2, 0
                if M[2, 2] > M[i, i]:
                    i, j, k = 2, 0, 1
                t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
                q[i] = t
                q[j] = M[i, j] + M[j, i]
                q[k] = M[k, i] + M[i, k]
                q[3] = M[k, j] - M[j, k]
                q = q[[3, 0, 1, 2]]
            q *= 0.5 / np.sqrt(t * M[3, 3])
        else:
            m00 = M[0, 0]
            m01 = M[0, 1]
            m02 = M[0, 2]
            m10 = M[1, 0]
            m11 = M[1, 1]
            m12 = M[1, 2]
            m20 = M[2, 0]
            m21 = M[2, 1]
            m22 = M[2, 2]
            # symmetric matrix K
            K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                             [m01+m10,     m11-m00-m22, 0.0,         0.0],
                             [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                             [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
            K /= 3.0
            # quaternion is eigenvector of K that corresponds to largest eigenvalue
            w, V = np.linalg.eigh(K)
            q = V[[3, 0, 1, 2], np.argmax(w)]
        if q[0] < 0.0:
            np.negative(q, q)
        return q
    
    def quat_to_rot(q):
        ''' Calculate rotation matrix corresponding to quaternion   
            q : 4 element quaternion
            M : (3,3) array
            Rotation matrix corresponding to input quaternion 
        '''
        w, x, y, z = q
        Nq = w*w + x*x + y*y + z*z
        if Nq < np.finfo(np.float).eps:
            return np.eye(3)
        s = 2.0/Nq
        X = x*s
        Y = y*s
        Z = z*s
        wX = w*X; wY = w*Y; wZ = w*Z
        xX = x*X; xY = x*Y; xZ = x*Z
        yY = y*Y; yZ = y*Z; zZ = z*Z
        return np.array(
               [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
                [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
                [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])
    
    def rotmat2axang(matrix):
    
        """Convert the rotation matrix into the axis-angle notation.
           The result is consistent with matlab implementation vrrotmat2vec
        Conversion equations
        ====================
            x = Qzy-Qyz
            y = Qxz-Qzx
            z = Qyx-Qxy
            r = hypot(x,hypot(y,z))
            t = Qxx+Qyy+Qzz
            theta = atan2(r,t-1)
        @param matrix:  The 3x3 rotation matrix to update.
        @type matrix:   3x3 numpy array
        @return:    The 3D rotation axis and angle.
        @rtype:     numpy 3D rank-1 array, float
        """
    
        # Axes.
        axis = np.zeros(3, np.float64)
        axis[0] = matrix[2,1] - matrix[1,2]
        axis[1] = matrix[0,2] - matrix[2,0]
        axis[2] = matrix[1,0] - matrix[0,1]
    
        # Angle.
        r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
        t = matrix[0,0] + matrix[1,1] + matrix[2,2]
        theta = math.atan2(r, t-1)
    
        # Normalise the axis.
        axis = axis / r
    
        # Return the data.
        return axis, theta        

    def vec2rotmat(angle, axis, point=None):
        """Return matrix to rotate about axis defined by point and axis.
        """
        sina = math.sin(angle)
        cosa = math.cos(angle)
        axis = Tools.__unit_vector(axis[:3])
        # rotation matrix around unit vector
        R = np.diag([cosa, cosa, cosa])
        R += np.outer(axis, axis) * (1.0 - cosa)
        axis *= sina
        R += np.array([[ 0.0,         -axis[2],  axis[1]],
                          [ axis[2], 0.0,          -axis[0]],
                          [-axis[1], axis[0],  0.0]])
        M = np.identity(3)
        M[:3, :3] = R
        if point is not None:
            M = np.identity(4)
            M[:3, :3] = R
            # rotation not around origin
            point = np.array(point[:3], dtype=np.float64, copy=False)
            M[:3, 3] = point - np.dot(R, point)
        return M
        
        
        
    def __vec_normalize(vec):
        eps=0.00001
        norm_vec=np.linalg.norm(vec)
        if (norm_vec<eps):
            vec_n=np.zeros(vec.size())
        else:
            vec_n=vec/norm_vec
        return vec_n
    
    def __unit_vector(data, axis=None, out=None):
        """Return ndarray normalized by length, i.e. Euclidean norm, along axis.
        """
        if out is None:
            data = np.array(data, dtype=np.float64, copy=True)
            if data.ndim == 1:
                data /= math.sqrt(np.dot(data, data))
                return data
        else:
            if out is not data:
                out[:] = np.array(data, copy=False)
            data = out
        length = np.atleast_1d(np.sum(data*data, axis))
        np.sqrt(length, length)
        if axis is not None:
            length = np.expand_dims(length, axis)
        data /= length
        if out is None:
            return data
        
    def random_quaternion(rand=None):
        """Return uniform random unit quaternion.   
        """
        if rand is None:
            rand = np.random.rand(3)
        else:
            assert len(rand) == 3
        r1 = np.sqrt(1.0 - rand[0])
        r2 = np.sqrt(rand[0])
        pi2 = math.pi * 2.0
        t1 = pi2 * rand[1]
        t2 = pi2 * rand[2]
        return np.array([math.cos(t2)*r2, math.sin(t1)*r1,
                            math.cos(t1)*r1, math.sin(t2)*r2])


    def rotmat2axang_alt(mat):    
        """Return rotation angle and axis from rotation matrix.#wiki:rotationmatrix
        This alternative formulation useseigendecomposition of the rotation 
        matrix which yields the eigenvalues 1 and cos θ ± i sin θ
        The result is nconsistent with matlab implementation vrrotmat2vec
         R(v,θ)=R(−v,−θ)
        """
        R = np.array(mat, dtype=np.float64, copy=False)
        # axis: unit eigenvector of R corresponding to eigenvalue of 1
        L, W = np.linalg.eig(R.T)
        i = np.where(abs(np.real(L) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError('no unit eigenvector corresponding to eigenvalue 1')
        axis = np.real(W[:, i[-1]]).squeeze()
        # rotation angle depending on axis
        cosa = (np.trace(R) - 1.0) / 2.0
        if abs(axis[2]) > 1e-8:
            sina = (R[1, 0] + (cosa-1.0)*axis[0]*axis[1]) / axis[2]
        elif abs(axis[1]) > 1e-8:
            sina = (R[0, 2] + (cosa-1.0)*axis[0]*axis[2]) / axis[1]
        else:
            sina = (R[2, 1] + (cosa-1.0)*axis[1]*axis[2]) / axis[0]
        angle = math.atan2(sina, cosa)
        return axis, angle   

    def rotmat2vec(mat1, rot_type='proper'):
        """
        Create an axis-angle np.array from Rotation Matrix:
        ====================
    
        @param mat:  The nx3x3 rotation matrices to convert
        @type mat:   nx3x3 numpy array
    
        @param rot_type: 'improper' if there is a possibility of
                          having improper matrices in the input,
                          'proper' otherwise. 'proper' by default
        @type  rot_type: string ('proper' or 'improper')
    
        @return:    The 3D rotation axis and angle (ax_ang)
                    5 entries:
                       First 3: axis
                       4: angle
                       5: 1 for proper and -1 for improper
        @rtype:     numpy 5xn array
    
        """
        mat = np.copy(mat1)
        if mat.ndim == 2:
            if np.shape(mat) == (3, 3):
                mat = np.copy(np.reshape(mat, (1, 3, 3)))
            else:
                raise Exception('Wrong Input Type')
        elif mat.ndim == 3:
            if np.shape(mat)[1:] != (3, 3):
                raise Exception('Wrong Input Type')
        else:
            raise Exception('Wrong Input Type')
    
        msz = np.shape(mat)[0]
        ax_ang = np.zeros((5, msz))
    
        epsilon = 1e-12
        if rot_type == 'proper':
            ax_ang[4, :] = np.ones(np.shape(ax_ang[4, :]))
        elif rot_type == 'improper':
            for i in range(msz):
                det1 = np.linalg.det(mat[i, :, :])
                if abs(det1 - 1) < epsilon:
                    ax_ang[4, i] = 1
                elif abs(det1 + 1) < epsilon:
                    ax_ang[4, i] = -1
                    mat[i, :, :] = -mat[i, :, :]
                else:
                    raise Exception('Matrix is not a rotation: |det| != 1')
        else:
            raise Exception('Wrong Input parameter for rot_type')
    
    
    
        mtrc = mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2]
    
    
        ind1 = np.where(abs(mtrc - 3) <= epsilon)[0]
        ind1_sz = np.size(ind1)
        if np.size(ind1) > 0:
            ax_ang[:4, ind1] = np.tile(np.array([0, 1, 0, 0]), (ind1_sz, 1)).transpose()
    
    
        ind2 = np.where(abs(mtrc + 1) <= epsilon)[0]
        ind2_sz = np.size(ind2)
        if ind2_sz > 0:
            # phi = pi
            # This singularity requires elaborate sign ambiguity resolution
    
            # Compute axis of rotation, make sure all elements >= 0
            # real signs are obtained by flipping algorithm below
            diag_elems = np.concatenate((mat[ind2, 0, 0].reshape(ind2_sz, 1),
                                         mat[ind2, 1, 1].reshape(ind2_sz, 1),
                                         mat[ind2, 2, 2].reshape(ind2_sz, 1)), axis=1)
            axis = np.sqrt(np.maximum((diag_elems + 1)/2, np.zeros((ind2_sz, 3))))
            # axis elements that are <= epsilon are set to zero
            axis = axis*((axis > epsilon).astype(int))
    
            # Flipping
            #
            # The algorithm uses the elements above diagonal to determine the signs
            # of rotation axis coordinate in the singular case Phi = pi.
            # All valid combinations of 0, positive and negative values lead to
            # 3 different cases:
            # If (Sum(signs)) >= 0 ... leave all coordinates positive
            # If (Sum(signs)) == -1 and all values are non-zero
            #   ... flip the coordinate that is missing in the term that has + sign,
            #       e.g. if 2AyAz is positive, flip x
            # If (Sum(signs)) == -1 and 2 values are zero
            #   ... flip the coord next to the one with non-zero value
            #   ... ambiguous, we have chosen shift right
    
            # construct vector [M23 M13 M12] ~ [2AyAz 2AxAz 2AxAy]
            # (in the order to facilitate flipping):    ^
            #                                  [no_x  no_y  no_z ]
    
            m_upper = np.concatenate((mat[ind2, 1, 2].reshape(ind2_sz, 1),
                                      mat[ind2, 0, 2].reshape(ind2_sz, 1),
                                      mat[ind2, 0, 1].reshape(ind2_sz, 1)), axis=1)
    
            # elements with || smaller than epsilon are considered to be zero
            signs = np.sign(m_upper)*((abs(m_upper) > epsilon).astype(int))
    
            sum_signs = np.sum(signs, axis=1)
            t1 = np.zeros(ind2_sz,)
            tind1 = np.where(sum_signs >= 0)[0]
            t1[tind1] = np.ones(np.shape(tind1))
    
            tind2 = np.where(np.all(np.vstack(((np.any(signs == 0, axis=1) == False), t1 == 0)), axis=0))[0]
            t1[tind2] = 2*np.ones(np.shape(tind2))
    
            tind3 = np.where(t1 == 0)[0]
            flip = np.zeros((ind2_sz, 3))
            flip[tind1, :] = np.ones((np.shape(tind1)[0], 3))
            flip[tind2, :] = np.copy(-signs[tind2, :])
    
            t2 = np.copy(signs[tind3, :])
    
            shifted = np.column_stack((t2[:, 2], t2[:, 0], t2[:, 1]))
            flip[tind3, :] = np.copy(shifted + (shifted == 0).astype(int))
    
            axis = axis*flip
            ax_ang[:4, ind2] = np.vstack((axis.transpose(), np.pi*(np.ones((1, ind2_sz)))))
    
        ind3 = np.where(np.all(np.vstack((abs(mtrc + 1) > epsilon, abs(mtrc - 3) > epsilon)), axis=0))[0]
        ind3_sz = np.size(ind3)
        if ind3_sz > 0:
            phi = np.arccos((mtrc[ind3]-1)/2)
            den = 2*np.sin(phi)
            a1 = (mat[ind3, 2, 1]-mat[ind3, 1, 2])/den
            a2 = (mat[ind3, 0, 2]-mat[ind3, 2, 0])/den
            a3 = (mat[ind3, 1, 0]-mat[ind3, 0, 1])/den
            axis = np.column_stack((a1, a2, a3))
            ax_ang[:4, ind3] = np.vstack((axis.transpose(), phi.transpose()))
    
        return ax_ang[:4].squeeze(),ax_ang[-1].squeeze()
    
    def mat2euler(M, cy_thresh=None):
        ''' Discover Euler angle vector from 3x3 matrix
        Uses the conventions above.
        Parameters
        ----------
        M : array-like, shape (3,3)
        cy_thresh : None or scalar, optional
           threshold below which to give up on straightforward arctan for
           estimating x rotation.  If None (default), estimate from
           precision of input.
        Returns
        -------
        z : scalar
        y : scalar
        x : scalar
           Rotations in radians around z, y, x axes, respectively
        Notes
        -----
        If there was no numerical error, the routine could be derived using
        Sympy expression for z then y then x rotation matrix, (see
        ``eulerangles.py`` in ``derivations`` subdirectory)::
          [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
          [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
          [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
        This gives the following solutions for ``[z, y, x]``::
           z = atan2(-r12, r11)
           y = asin(r13)
           x = atan2(-r23, r33)
        Problems arise when ``cos(y)`` is close to zero, because both of::
           z = atan2(cos(y)*sin(z), cos(y)*cos(z))
           x = atan2(cos(y)*sin(x), cos(x)*cos(y))
        will be close to ``atan2(0, 0)``, and highly unstable.
        The ``cy`` fix for numerical instability in this code is from: *Euler Angle
        Conversion* by Ken Shoemake, p222-9 ; in: *Graphics Gems IV*, Paul Heckbert
        (editor), Academic Press, 1994, ISBN: 0123361559.  Specifically it comes
        from ``EulerAngles.c`` and deals with the case where cos(y) is close to
        zero:
        * http://www.graphicsgems.org/
        * https://github.com/erich666/GraphicsGems/blob/master/gemsiv/euler_angle/EulerAngles.c#L68
        The code appears to be licensed (from the website) as "can be used without
        restrictions".
        '''
        M = np.asarray(M)
        if cy_thresh is None:
            try:
                cy_thresh = np.finfo(M.dtype).eps * 4
            except ValueError:
                cy_thresh = _FLOAT_EPS_4
        r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
        # (-cos(y)*sin(x))**2 + (cos(x)*cos(y))**2) =
        # (cos(y)**2)(sin(x)**2 + cos(x)**2) ==> (Pythagoras)
        # cos(y) = sqrt((-cos(y)*sin(x))**2 + (cos(x)*cos(y))**2)
        cy = math.sqrt(r23 * r23 + r33 * r33)
        if cy > cy_thresh: # cos(y) not close to zero, standard form
            z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = math.atan2(r13,  cy) # atan2(sin(y), cy)
            x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else: # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = math.atan2(r21,  r22)
            y = math.atan2(r13,  cy) # atan2(sin(y), cy)
            x = 0.0
        return z, y, x
    

