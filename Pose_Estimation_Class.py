# -*- coding: utf-8 -*-
"""
    + Simultaneous Robot/World and Tool/Flange Calibration:    
    Implementation of Shah, Mili. "Solving the robot-world/hand-eye calibration problem using the Kronecker product." 
    Journal of Mechanisms and Robotics 5.3 (2013): 031007.
    
    Batch_Processing solvesfor  X and Y in AX=YB from a set of (A,B) paired measurements.
    (Ai,Bi) are absolute pose measurements with known correspondance       

    A: (4x4xn) 
    X: (4x4): unknown
    Y: (4x4): unknown
    B: (4x4xn) 
    n number of measurements
    
    + EKF,IEKF solves for AX=XB from a set of (Ai,Bi) relative pose measurements with known correspondance.
    so3 representation was used to represent the state of rotation.  
    
    @author: elif.ayvali
"""
from helpers import Tools
import pickle
import numpy as np
EPS=0.00001

class Batch_Processing:        
    def pose_estimation(A,B):
   
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
        X=np.matmul(U_n,Vt_n)
    
        Y=np.reshape(yp, (3,3), order="F")#F: fortran/matlab reshape order
        Yn = (np.sign(np.linalg.det(Y))/ np.abs(np.linalg.det(Y))**(1/3))*Y
        U_yn, S_yn, Vt_yn=np.linalg.svd(Yn)
        Y=np.matmul(U_yn,Vt_yn)
      
        A_est = np.zeros([3*n,6])
        b_est = np.zeros([3*n,1])
        for ii in range(n-1):       
            A_est[3*ii:3*ii+3,:] =np.concatenate((-A[0:3,0:3,ii], np.eye(3)),axis=1)         
            b_est[3*ii:3*ii+3,:] = np.transpose(A[0:3,3,ii] - np.matmul(np.kron(B[0:3,3,ii].T,np.eye(3)), np.reshape(Y, (9,1), order="F")).T)
    
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
           AX[:,:,ii]=np.matmul(A[:,:,ii],X)        
           
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
        U, S, Vt=np.linalg.svd(np.matmul(Bp_2D[0:3,:],AXp_2D[0:3,:].T))# v is v' in matlab  
        R_est = np.matmul(Vt.T, U.T)
        # special reflection case
        if np.linalg.det(R_est) < 0:
            print ('Warning: Y_est returned a reflection')
            R_est =np.matmul( Vt.T, np.matmul(np.diag([1,1,-1]),U.T))       
        #Calculates the best transformation
        t_est = t-np.dot(R_est,that)
        Y_est[0:3,0:3]=R_est
        Y_est[0:3,3]=t_est
        #Calculate registration error
        pYB=np.matmul(R_est,B[0:3,3,:])+np.tile(t_est[:,np.newaxis],(1,n))#3xn
        pAX=AX[0:3,3,:]
        Reg_error=np.linalg.norm(pAX-pYB,axis=0) #1xn
        ErrorStats[0]=np.mean(Reg_error)
        ErrorStats[1]=np.std(Reg_error)
        return Y_est, ErrorStats

class EKF(object):
    def __init__(self):
        
        self.x=np.array([0,1,0,0,0,0],dtype=np.float64)
        self.P=np.diag([1.0,1.0,1.0,1.0,1.0,1.0])
        self.R=np.diag([1.0,1.0,1.0,1.0,1.0,1.0])  #if zero S,P grows
        self.z=np.zeros(6,dtype=np.float64) #pseudo measurements
        self.consistency=[] #should decrease over time for the LSE problem
        
    def Update(self,AA,BA):
        #process model is constant so no prediction step
        h=self.__CalculateMeasurementFunction(self.x, AA, BB)
        H=self.__CalculateJacobian(self.x,AA,BB)  
        S=np.linalg.multi_dot([H,self.P,H.T])+self.R
        K =np.linalg.multi_dot([self.P, H.T,np.linalg.inv(S)])
      
        y=self.z-h
        self.x=self.x+np.dot(K,y)
        self.P=np.matmul(np.identity(np.size(self.x))-np.matmul(K,H), self.P)

        #consistency check (NIS, dof 6) xi-squared
        self.consistency.append(np.linalg.multi_dot([y.T,np.linalg.inv(S),y]))
        
    def __CalculateJacobian(self,x, AA,BB):
        h0=self.__CalculateMeasurementFunction(x, AA, BB)
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
        else:
            k=x[:3]/np.linalg.norm(x[:3])
            
        Rx=Tools.vec2rotmat(theta, k)
        v_AAX,_=Tools.rotmat2vec(np.matmul(AA[:3,:3],Rx))
        v_XBB,_=Tools.rotmat2vec(np.matmul(Rx,BB[:3,:3])) #axis,angle
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
        self.P=np.diag([1.0,1.0,1.0,1.0,1.0,1.0])
        self.R=np.diag([1.0,1.0,1.0,1.0,1.0,1.0])  #if zero S,P grows
        self.z=np.zeros(6,dtype=np.float64) #pseudo measurements
        self.consistency=[] #should decrease over time for the LSE problem
        
    def Update(self,AA,BA):
        #process model is constant so no prediction step
        numIterations=0
        maxIterations=5
        innovation=0
        stop_thresh=0.02 #first implement EKF then monitor this variable to tune
        iterations_done=False
        xi=np.copy(self.x)
        
        while numIterations<maxIterations and iterations_done==False:      
            hi=self.__CalculateMeasurementFunction(xi, AA, BB)
            Hi=self.__CalculateJacobian(xi,AA,BB)  
            Si=np.linalg.multi_dot([Hi,self.P,Hi.T])+self.R
            Ki =np.linalg.multi_dot([self.P, Hi.T,np.linalg.inv(Si)])
            yi=self.z-hi-np.dot(Hi,self.x-xi)
            xi=self.x+np.dot(Ki,yi)   
            numIterations=numIterations+1
            innovation =np.linalg.norm(yi)             
            #another criteria: x_diff=np.norm(self.x-xi)/np.norm(self.x) 
            #use relative err for floating point 
            if innovation<stop_thresh: 
                iterations_done=True    
        #consistency check (NIS, dof 6) xi-squared
        self.consistency.append(np.linalg.multi_dot([yi.T,np.linalg.inv(Si),yi]))
        #Update state and covariance
        self.x=np.copy(xi)
        H= self.__CalculateJacobian(self.x,AA,BB)  
        S=np.linalg.multi_dot([H,self.P,H.T])+self.R
        K =np.linalg.multi_dot([self.P, H.T,np.linalg.inv(S)])        
        self.P=np.matmul(np.identity(np.size(self.x))-np.matmul(K,H), self.P)


        
    def __CalculateJacobian(self,x, AA,BB):
        h0=self.__CalculateMeasurementFunction(x, AA, BB)
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
        else:
            k=x[:3]/np.linalg.norm(x[:3])
            
        Rx=Tools.vec2rotmat(theta, k)
        v_AAX,_=Tools.rotmat2vec(np.matmul(AA[:3,:3],Rx))
        v_XBB,_=Tools.rotmat2vec(np.matmul(Rx,BB[:3,:3])) #axis,angle
        h[:3]=v_AAX[:3]-v_XBB[:3]
        #Ratx+ta-Rxtb-tx
        ta=AA[0:3,3]
        tb=BB[0:3,3]
        tx=x[3:6]
        h[3:]=np.dot(AA[:3,:3],tx)+ta-np.dot(Rx,tb)-tx
        return h


class UKF(object): 
    def __init__(self):        
        self.x=np.array([0,1,0,0,0,0],dtype=np.float64)
        self.nx=self.x.size
        self.num_sigma=2*self.nx+1
        self.l=6-self.nx #change and easily see its affect
        self.P=np.diag([1.0,1.0,1.0,1.0,1.0,1.0])
        self.R=np.diag([1.0,1.0,1.0,1.0,1.0,1.0])  #if zero S,P grows
        self.z=np.zeros(self.nx,dtype=np.float64) #pseudo measurements
        self.L=np.diag([1.0,1.0,1.0,1.0,1.0,1.0])
        self.sigma_pts=np.zeros((self.nx,self.num_sigma))
        self.weights=np.zeros((self.num_sigma,1))
        self.update_thresh=0.02 #same as IEKF stop thresh
        self.consistency=[] #should decrease over time for the LSE problem
        
    def Update(self,AA,BA):
        h=np.zeros((self.nx,self.num_sigma),dtype=np.float64) #measurements
        h_mean=np.zeros(self.nx,dtype=np.float64) #measurements
        #process model is constant so no prediction step
        w,V=np.linalg.eig(self.P)
        self.P=np.linalg.multi_dot([V,np.diag(w),V.T])
        #make sure P is pos def for cholesky decomposition
        self.P=Tools.nearestPSD(np.copy(self.P))
        self.L=np.linalg.cholesky(self.P)#lower-triangular
        self.__GenerateSigmaPoints()
        self.__SetWeights()
        for i in range(self.num_sigma):
            h[:,i]=self.__CalculateMeasurementFunction(self.sigma_pts[:,i],AA,BB)
            h_mean=np.copy(h_mean)+self.weights[i]*h[:,i]
        S=np.copy(self.R)
        T=np.zeros_like(self.P)
        for i in range(self.num_sigma):
            h_diff=h[:,i]-h_mean
            S+=self.weights[i]*np.outer(h_diff,h_diff.T)
            T+=self.weights[i]*np.outer(self.sigma_pts[:,i].T,h_diff.T)   
        #careful with Sinv   
        K=np.matmul(T,np.linalg.inv(S))
        y=self.z-h_mean
        innovation =np.linalg.norm(y) 
#        if innovation>self.update_thresh:
        self.P-=np.linalg.multi_dot([K,S,K.T])
        self.x=self.x+np.dot(K,y)
        #consistency check (NIS, dof 6) xi-squared
        self.consistency.append(np.linalg.multi_dot([y.T,np.linalg.inv(S),y]))
    

    def __GenerateSigmaPoints(self):
        #set first column of sigma point matrix
        self.sigma_pts[:,0]=self.x
        for i in range(self.nx):
            self.sigma_pts[:,i+1]=self.x+np.dot(np.sqrt(self.l+self.nx),self.L[:,i])
            self.sigma_pts[:,i+1+self.nx]=self.x-np.dot(np.sqrt(self.l+self.nx),self.L[:,i] )
            
    def __SetWeights(self):   
        #set the first weight
        self.weights[0]=self.l/(self.l+self.nx)
        for i in range(self.num_sigma-1):
            self.weights[i+1]=0.5/(self.l+self.nx)            
            
    def __CalculateMeasurementFunction(self, x, AA, BB):
        h=np.zeros(6)
        theta=np.linalg.norm(x[0:3])
        if theta < EPS:
           k=[0,1,0] #VRML standard
        else:
            k=x[:3]/np.linalg.norm(x[:3])
            
        Rx=Tools.vec2rotmat(theta, k)
        v_AAX,_=Tools.rotmat2vec(np.matmul(AA[:3,:3],Rx))
        v_XBB,_=Tools.rotmat2vec(np.matmul(Rx,BB[:3,:3])) #axis,angle
        h[:3]=v_AAX[:3]-v_XBB[:3]
        #Ratx+ta-Rxtb-tx
        ta=AA[0:3,3]
        tb=BB[0:3,3]
        tx=x[3:6]
        h[3:]=np.dot(AA[:3,:3],tx)+ta-np.dot(Rx,tb)-tx
        return h
              
        
data_file='pose_sim_data.p'#random 3deg, 3mm noise added to measurements
with open(data_file, mode='rb') as f:
    sim_data = pickle.load(f)
A_seq=sim_data['xfm_A']
B_seq=sim_data['xfm_B']
AA_seq=sim_data['xfm_AA']
BB_seq=sim_data['xfm_BB']
X=sim_data['X']
Y=sim_data['Y']


#Ground Truth
print('\n')
print('.....Ground Truth')
euler_GT=Tools.mat2euler(X[:3,:3])
print("GT[euler_rpy(deg) , pos(mm)]:",np.array(euler_GT)*180/np.pi,X[:3,3].T*100)

#Batch Processing
X_est,Y_est,Y_est_check,ErrorStats=Batch_Processing.pose_estimation(A_seq,B_seq)
print('\n')
print('.....Batch Processing Results')
euler_batch=Tools.mat2euler(X_est[:3,:3])
batch_euler_err=np.array(euler_batch)*180/np.pi-np.array(euler_GT)*180/np.pi
batch_pos_err=X_est[:3,3].T*100-X[:3,3].T*100
print("Batch[euler_rpy(deg) , pos(mm)]:",np.array(euler_batch)*180/np.pi,X_est[:3,3].T*100)
print("Error[euler_rpy(deg) , pos(mm)]:", batch_euler_err, batch_pos_err)

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
print('\n')
print('.....EKF Results')
ekf_euler_err=np.array(euler_ekf)*180/np.pi-np.array(euler_GT)*180/np.pi
ekf_pos_err=ekf.x[3:].T*100-X[:3,3].T*100
print("EKF  [euler_rpy(deg) , pos(mm)]:",np.array(euler_ekf)*180/np.pi,ekf.x[3:]*100)
print("Error[euler_rpy(deg) , pos(mm)]:", ekf_euler_err, ekf_pos_err)



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
euler_iekf=Tools.mat2euler(Tools.vec2rotmat(theta, k))

print('\n')
print('.....IEKF Results')

iekf_euler_err=np.array(euler_iekf)*180/np.pi-np.array(euler_GT)*180/np.pi
iekf_pos_err=iekf.x[3:].T*100-X[:3,3].T*100
print("IEKF [euler_rpy(deg) , pos(mm)]:",np.array([euler_iekf])*180/np.pi,iekf.x[3:]*100)
print("Error[euler_rpy(deg) , pos(mm)]:", iekf_euler_err, iekf_pos_err)

#UKF
ukf=UKF()
for i in range(len(AA_seq[1,1,:])):
    AA=AA_seq[:,:,i] 
    BB=BB_seq[:,:,i]
    ukf.Update(AA,BB)
    
theta=np.linalg.norm(ukf.x[:3])
if theta < EPS:
   k=[0,1,0] #VRML standard
else:
    k=ukf.x[0:3]/np.linalg.norm(ukf.x[:3])
euler_ukf=Tools.mat2euler(Tools.vec2rotmat(theta, k))
print('\n')
print('.....UKF Results')

ukf_euler_err=np.array(euler_ukf)*180/np.pi-np.array(euler_GT)*180/np.pi
ukf_pos_err=ukf.x[3:].T*100-X[:3,3].T*100
print("UKF [euler_rpy(deg) , pos(mm)]:",np.array([euler_ukf])*180/np.pi,ukf.x[3:]*100)
print("Error[euler_rpy(deg) , pos(mm)]:", ukf_euler_err, ukf_pos_err)


