from collections import deque
# from matplotlib import pyplot
import numpy as np
from math import *
# from mpl_toolkits.mplot3d import Axes3D
import copy

class MeasurePoint:
    def __init__(self,theta=0,w=0,
                typ=None,confidence=0.0):
        self.theta,self.w=theta,w
        self.typ=typ
        self.confidence=confidence

    def __str__(self):
        return '--theta={0},w={1}'.format(self.theta,self.w)

def norm_pdf_multivariate(x, mu, sigma):
    x,mu=np.array(x),np.array(mu)
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( pow((2*pi),float(size)/2) * pow(det,1.0/2) )
        x_mu = np.array(x - mu).reshape(1,size)
        inv = np.linalg.inv(sigma)
        result = pow(e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

class Track:
    """
    Track which include measure point up to "maxLen" frames ago.

    The measure point at time k is predicted if frame(k) disappear.

    Parameters
    -------------
    pStart: MeasurePoint
        use start point to initialize.

    T: int
        predict time.
    """
    def __init__(self,pStart,T,idTrack):
        self.maxLen=1
        self.T=T
        self.trackPoint=deque(maxlen=self.maxLen)
        self.trackPoint.append(pStart)
        self.state=np.array([pStart.theta,0]).reshape([2,1])
        self.P=10*np.identity(2)

        self.idTrack = idTrack

        self.F=np.array([[1,T],
                        [0,1]])
        self.Q=np.array([[T**4/4,T**3/2],
                        [T**3/2,T**2]])*3

        self.H=np.array([[1,0]])
        self.R=np.array([[15]])

    def AddPoint(self,p):
        """
        Add one point to track.

        Parameters
        --------------
        p: MeasurePoint
            The measurement point in this frame.
        """
        self.PredictNextPos()
        self.update(p)
        ptmp=copy.deepcopy(p)
        ptmp.theta,ptmp.w=self.state[0][0],self.state[1][0]
        if self.trackPoint[-1].confidence>ptmp.confidence:
            ptmp.confidence=self.trackPoint[-1].confidence
            ptmp.typ=self.trackPoint[-1].typ
        self.trackPoint.append(ptmp)
    
    def PredictNextPos(self):
        """
        Predict next point's position using Kalman Filter and CV model.
        
        Returns
        ----------
        predicted point: MeasurePoint
            The predicted point.
        """
        self.state_=np.matmul(self.F,self.state) #预测状态均值
        self.P_=np.matmul(np.matmul(self.F,self.P),self.F.T)+self.Q #预测状态方差
        self.Z_=np.matmul(self.H,self.state_) #获取预测状态中角度
        ptmp=copy.deepcopy(self.trackPoint[-1])
        ptmp.theta=self.Z_[0][0]     #将最后一个测量点的角度进行预测
        #print('  --track.predict finished')
        return ptmp

    def update(self,p):
        """
        Update state matrix and state covariance matrix.

        Parameters
        -------------
        p: MeasurePoint
            Measure point of this frame.
            Equal to predicted point if no match point is found.
        """
        self.Z=np.array([[p.theta]]).T
        self.v=self.Z-self.Z_
        self.S=np.matmul(np.matmul(self.H,self.P_),self.H.T)+self.R
        self.K=np.matmul(np.matmul(self.P_,self.H.T),np.linalg.inv(self.S)) #卡尔曼增益
        self.state=self.state_+np.matmul(self.K,self.v)#状态更新（状态估计）
        self.P=self.P_-np.matmul(self.K,np.matmul(self.S,self.K.T))#状态协方差更新
        #print('  ----track.update finished')

    def ingate(self,p,thres=1):
        """
        Judge whether p is in gate.

        if:
            (z-Hx_)^T.(H.P_.H^T+R).(z-Hx_)<=thres
        return true

        Parameters
        -------------
        p: MeasurePoint
        
        thres: float (0<thres)
        """
        #print('    track.ingate!!!')
        self.PredictNextPos()
        delta=np.array([[p.theta]])-self.Z_    #测量值与预测值的差
        B=np.dot(np.dot(self.H,self.P_),self.H.T)+self.R
        return np.dot(np.dot(delta.T,np.linalg.inv(B)),delta)<=thres

    def estimateP(self,p):
        """
        Estimate the probability of generating p from this track.

        N(z-Hx_;0,H.P_.H^T+R)

        Parameters
        -------------
        p: MeasurePoint
        
        thres: float (0<thres)
        """
        self.PredictNextPos()
        B=np.dot(np.dot(self.H,self.P_),self.H.T)+self.R
        ret=norm_pdf_multivariate([p.theta-self.Z_[0][0]],[0],B)
        return ret

# if __name__=='__main__':
    ############# KF ####################
    # p=MeasurePoint(0,0,0)
    # tk=Track(p,0.1)
    # all=50

    # fig = pyplot.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # for t in range(1,all):
    #     p.x=cos(pi/2*t*0.1)
    #     p.y=sin(pi/2*t*0.1)
    #     p.z=0.1*t*0.1
    #     ax.scatter(p.x,p.y,p.z,c='b')
    #     tk.AddPoint(p)
    #     ax.scatter(tk.trackPoint[-1].x,tk.trackPoint[-1].y,tk.trackPoint[-1].z,c='r')

    # pyplot.show()