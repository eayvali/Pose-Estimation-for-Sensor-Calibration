# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:37:36 2019

@author: elif.ayvali
"""
from helpers import Tools
import pickle
import numpy as np
EPS=0.00001

class Batch_Processing:        
    def pose_estimation(A,B):
        """solves 
        A: (4x4xn) 
        X: (4x4): unknown
        y: (4x4): unknown
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

class EKF(object):
    def __init__(self):
        
        self.x=np.array([0,1,0,0,0,0],dtype=np.float64)
        self.P=np.diag([1.0,1.0,1.0,100.0,100.0,100.0])
        self.R=np.diag([1.0,1.0,1.0,1.0,1.0,1.0])  
        self.z=np.zeros(6,dtype=np.float64) #pseudo measurements
        
    def Update(self,AA,BA):
        #process model is constant so no prediction step
        h=self.__CalculateMeasurementFunction(self.x, AA, BB)
        H=self.__CalculateJacobian(self.x,AA,BB)  
        S=np.linalg.multi_dot([H,self.P,H.T])+self.R
        K =np.linalg.multi_dot([self.P, H.T,np.linalg.inv(S)])
        
        y=self.z-h
        self.x=self.x+np.dot(K,y)
        self.P=np.dot(np.identity(np.size(self.x))-np.dot(K,H), self.P)

        
    def __CalculateJacobian(self,x, AA,BB):
        h0=self.__CalculateMeasurementFunction(x, AA, BB)
#        print("h0",h0)
        H=np.zeros((np.size(h0),np.size(self.x)))     
        dt=np.float64(0.001)
        for i in range(len(x)):
            x_temp=np.copy(x)
            x_temp[i]=x_temp[i]+dt     
            H[:,i]=(self.__CalculateMeasurementFunction(x_temp,AA,BB)-h0)/dt;#row_vec
        return H 
        
    def __CalculateMeasurementFunction(self, x, AA, BB):
        h=np.zeros(6)
        theta=np.linalg.norm(x[0:3])
        if theta < EPS:
           k=[0,1,0] #VRML standard
           # k=[0,0,1]# ISO/IEC IS 19775-1:2013 standard
        else:
            k=x[:3]/np.linalg.norm(x[:3])
            
        Rx=Tools.vec2rotmat(theta, k)
        v_AAX,_=Tools.rotmat2vec(np.dot(AA[:3,:3],Rx))
        v_XBB,_=Tools.rotmat2vec(np.dot(Rx,BB[:3,:3])) #axis,angle
        h[:3]=v_AAX[:3]-v_XBB[:3]
        #Ratx+ta-Rxtb-tx
        ta=AA[0:3,3]
        tb=BB[0:3,3]
        tx=x[3:6]
        h[3:]=np.dot(AA[:3,:3],tx)+ta-np.dot(Rx,tb)-tx
        return h
           
class IEKF(object):
    def __init__(self):
        
        self.x=np.array([0,1,0,0,0,0],dtype=np.float64)
        self.P=np.diag([1.0,1.0,1.0,100.0,100.0,100.0])
        self.R=np.diag([1.0,1.0,1.0,1.0,1.0,1.0])  
        self.z=np.zeros(6,dtype=np.float64) #pseudo measurements
        
    def Update(self,AA,BA):
        #process model is constant so no prediction step
        numIterations=0
        maxIterations=10
        innovation=0
        stop_thresh=0.01 #needs to be tuned
        iterations_done=False
        xi=np.copy(self.x)
        
        while numIterations<maxIterations and iterations_done==False:      
            hi=self.__CalculateMeasurementFunction(xi, AA, BB)
            Hi=self.__CalculateJacobian(xi,AA,BB)  
            Si=np.linalg.multi_dot([Hi,self.P,Hi.T])+self.R
            Ki =np.linalg.multi_dot([self.P, Hi.T,np.linalg.inv(Si)])
            
            yi=self.z-hi-np.dot(Hi,(self.x-xi))
            xi=self.x+np.dot(Ki,yi)   
            numIterations=numIterations+1
            innovation =np.linalg.norm(yi)          
            if innovation<stop_thresh:
                iterations_done=True                      
        self.x=np.copy(xi)
        H= self.__CalculateJacobian(self.x,AA,BB)  
        S=np.linalg.multi_dot([H,self.P,H.T])+self.R
        K =np.linalg.multi_dot([self.P, H.T,np.linalg.inv(S)])        
        self.P=np.dot(np.identity(np.size(self.x))-np.dot(K,H), self.P)

        
    def __CalculateJacobian(self,x, AA,BB):
        h0=self.__CalculateMeasurementFunction(x, AA, BB)
#        print("h0",h0)
        H=np.zeros((np.size(h0),np.size(x)))     
        dt=np.float64(0.001)
        for i in range(len(x)):
            x_temp=np.copy(x)
            x_temp[i]=x_temp[i]+dt     
            H[:,i]=(self.__CalculateMeasurementFunction(x_temp,AA,BB)-h0)/dt;#row_vec
        return H 
        
    def __CalculateMeasurementFunction(self, x, AA, BB):
        h=np.zeros(6)
        theta=np.linalg.norm(x[0:3])
        if theta < EPS:
           k=[0,1,0] #VRML standard
           # k=[0,0,1]# ISO/IEC IS 19775-1:2013 standard
        else:
            k=x[:3]/np.linalg.norm(x[:3])
            
        Rx=Tools.vec2rotmat(theta, k)
        v_AAX,_=Tools.rotmat2vec(np.dot(AA[:3,:3],Rx))
        v_XBB,_=Tools.rotmat2vec(np.dot(Rx,BB[:3,:3])) #axis,angle
        h[:3]=v_AAX[:3]-v_XBB[:3]
        #Ratx+ta-Rxtb-tx
        ta=AA[0:3,3]
        tb=BB[0:3,3]
        tx=x[3:6]
        h[3:]=np.dot(AA[:3,:3],tx)+ta-np.dot(Rx,tb)-tx
        return h
        
data_file='pose_sim_data_noisy.p'
with open(data_file, mode='rb') as f:
    sim_data = pickle.load(f)
#Xnoise=[ 3*(2*rand-1)/100, 3*(2*rand-1)/100, 3*(2*rand-1)/100,3*(2*rand-1)*pi/180,3*(2*rand-1)*pi/180,3*(2*rand-1)*pi/180]
A_seq=sim_data['xfm_A']
B_seq=sim_data['xfm_B']
AA_seq=sim_data['xfm_AA']
BB_seq=sim_data['xfm_BB']
X=sim_data['X']
Y=sim_data['Y']


#Ground Truth
print('.....Ground Truth')
euler_GT=Tools.mat2euler(X[:3,:3])
print("GT[euler_rpy(deg) , pos(mm)]:",np.array([euler_GT])*180/np.pi,X[:3,3].T*100)

#Batch Processing
X_est,Y_est,Y_est_check,ErrorStats=Batch_Processing.pose_estimation(A_seq,B_seq)
print('.....Batch Processing Results')
euler_batch=Tools.mat2euler(X_est[:3,:3])
print("Batch[euler_rpy(deg) , pos(mm)]:",np.array([euler_batch])*180/np.pi,X_est[:3,3].T*100)

#EKF
ekf=EKF()
for i in range(len(AA_seq[1,1,:])):
    AA=AA_seq[:,:,i] 
    BB=BB_seq[:,:,i]
    ekf.Update(AA,BB)
    
theta=np.linalg.norm(ekf.x[:3])
if theta < EPS:
   k=[0,1,0] #VRML standard
else:
    k=ekf.x[0:3]/np.linalg.norm(ekf.x[:3])
euler_ekf=Tools.mat2euler(Tools.vec2rotmat(theta, k))
print('.....EKF Results')
print("EKF[euler_rpy(deg) , pos(mm)]:",np.array([euler_ekf])*180/np.pi,ekf.x[3:]*100)


#IEKF
iekf=IEKF()
for i in range(len(AA_seq[1,1,:])):
    AA=AA_seq[:,:,i] 
    BB=BB_seq[:,:,i]
    iekf.Update(AA,BB)
    
theta=np.linalg.norm(iekf.x[:3])
if theta < EPS:
   k=[0,1,0] #VRML standard
else:
    k=iekf.x[0:3]/np.linalg.norm(iekf.x[:3])
euler_ekf=Tools.mat2euler(Tools.vec2rotmat(theta, k))
print('.....IEKF Results')
print("IEKF[euler_rpy(deg) , pos(mm)]:",np.array([euler_ekf])*180/np.pi,iekf.x[3:]*100)




