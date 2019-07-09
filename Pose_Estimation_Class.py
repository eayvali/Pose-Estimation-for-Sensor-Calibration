# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:37:36 2019

@author: elif.ayvali
"""

import pickle
import numpy as np

class Batch_Processing:        
    def pose_estimation(A,B):
        """solves 
        A: (4x4xn) 
        X: (4x4)   
        B: (4x4xn) 
        n number of measurements
        (Ai,Bi) has known correspondance
        Implementation of Shah, Mili. "Solving the robot-world/hand-eye calibration problem using the Kronecker product." 
        Journal of Mechanisms and Robotics 5.3 (2013): 031007.
        Simultaneous Robot/World and Tool/Flange 
        Calibration by Solving for  X and Y in AX=YB
        """    
        n=A.shape[2];
        T = np.zeros([9,9]);
        X_est= np.eye(4)
        Y_est= np.eye(4)

        #Permutate A and B to get gross motions
        idx = np.random.permutation(n)
        A=A[:,:,idx];
        B=B[:,:,idx];
    
        for ii in range(n-1):   
            Ra = A[0:3,0:3,ii]
            Rb = B[0:3,0:3,ii]
          #  K[9*ii:9*(ii+1),:] = np.concatenate((np.kron(Rb,Ra), -np.eye(9)),axis=1)
            T = T + np.kron(Rb,Ra);
        
        U, S, Vt=np.linalg.svd(T)
        xp=Vt.T[:,0]
        yp=U[:,0]
        X=np.reshape(xp, (3,3), order="F")#F: fortran/matlab reshape order
        Xn = (np.sign(np.linalg.det(X))/ np.abs(np.linalg.det(X))**(1/3))*X
        #re-orthogonalize to guarantee that they are indeed rotations.
        U_n, S_n, Vt_n=np.linalg.svd(Xn)
        X=np.dot(U_n,Vt_n)
    
        Y=np.reshape(yp, (3,3), order="F")#F: fortran/matlab reshape order
        Yn = (np.sign(np.linalg.det(Y))/ np.abs(np.linalg.det(Y))**(1/3))*Y
        U_yn, S_yn, Vt_yn=np.linalg.svd(Yn)
        Y=np.dot(U_yn,Vt_yn)
      
        A_est = np.zeros([3*n,6])
        b_est = np.zeros([3*n,1])
        for ii in range(n-1):       
            A_est[3*ii:3*ii+3,:] =np.concatenate((-A[0:3,0:3,ii], np.eye(3)),axis=1)         
            b_est[3*ii:3*ii+3,:] = np.transpose(A[0:3,3,ii] - np.dot(np.kron(B[0:3,3,ii].T,np.eye(3)), np.reshape(Y, (9,1), order="F")).T)
    
        t_est_np=np.linalg.lstsq(A_est,b_est,rcond=None)
        if t_est_np[2]<A_est.shape[1]: # A_est.shape[1]=6
            print('Rank deficient')
        t_est=t_est_np[0]
        X_est[0:3,0:3]= X
        X_est[0:3,3]= t_est[0:3].T  
        Y_est[0:3,0:3]= Y    
        Y_est[0:3,3]= t_est[3:6].T        
        #verify Y_est using rigid_registration
        Y_est_check,ErrorStats= Batch_Processing.__rigid_registration(A,X_est,B)
        return X_est,Y_est, Y_est_check,ErrorStats
    
    def __rigid_registration(A,X,B):
        #nxnx4            
        """solves for Y in YB=AX
        A: (4x4xn) 
        B: (4x4xn) 
        X= (4x4)   
        Y= (4x4)       
        n number of measurements
        ErrorStats: Registration error (mean,std)
        """
        n=A.shape[2];
        AX=np.zeros(A.shape)
        AXp=np.zeros(A.shape)
        Bp=np.zeros(B.shape)
        pAX=np.zeros(B[0:3,3,:].shape)#To calculate reg error    
        pYB=np.zeros(B[0:3,3,:].shape)#To calculate reg error  
        Y_est=np.eye(4)

        ErrorStats=np.zeros((2,1))
        
        for ii in range(n):
           AX[:,:,ii]=np.dot(A[:,:,ii],X)        
           
        #Centroid of transformations t and that
        t=1/n*np.sum(AX[0:3,3,:],1);
        that=1/n*np.sum(B[0:3,3,:],1);
        AXp[0:3,3,:]=AX[0:3,3,:]-np.tile(t[:,np.newaxis], (1, n))
        Bp[0:3,3,:]=B[0:3,3,:]-np.tile(that[:,np.newaxis], (1, n))

        [i,j,k]=AX.shape; #4x4xn
        #Convert AX and B to 2D arrays
        AXp_2D=AXp.reshape((i,j*k)) # now it is 4x(4xn)
        Bp_2D=Bp.reshape((i,j*k))# 4x(4xn)        
        #%Calculates the best rotation
        U, S, Vt=np.linalg.svd(np.dot(Bp_2D[0:3,:],AXp_2D[0:3,:].T))# v is v' in matlab  
        R_est = np.dot(Vt.T, U.T)
        # special reflection case
        if np.linalg.det(R_est) < 0:
            print ('Warning: Y_est returned a reflection')
            R_est =np.dot( Vt.T, np.dot(np.diag([1,1,-1]),U.T))       
        #Calculates the best transformation
        t_est = t-np.dot(R_est,that)
        Y_est[0:3,0:3]=R_est
        Y_est[0:3,3]=t_est
        #Calculate registration error
        pYB=np.dot(R_est,B[0:3,3,:])+np.tile(t_est[:,np.newaxis],(1,n))#3xn
        pAX=AX[0:3,3,:]
        Reg_error=np.linalg.norm(pAX-pYB,axis=0) #1xn
        ErrorStats[0]=np.mean(Reg_error)
        ErrorStats[1]=np.std(Reg_error)
        return Y_est, ErrorStats

        
data_file='pose_sim_data_noisy.p'
with open(data_file, mode='rb') as f:
    sim_data = pickle.load(f)

A_seq=sim_data['A']
B_seq=sim_data['B']
X=sim_data['X']
Y=sim_data['Y']

X_est,Y_est,Y_est_check,ErrorStats=Batch_Processing.pose_estimation(A_seq,B_seq)
print(X_est,Y_est,Y_est_check,ErrorStats)


