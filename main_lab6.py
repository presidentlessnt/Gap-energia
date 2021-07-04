#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import modulo as mtd
import scipy as sci
from scipy.optimize import curve_fit
from scipy import odr


## Obtener datos del .txt ##

SiA=np.loadtxt('Si_A.txt')
SiB=np.loadtxt('Si_B.txt')
GeA=np.loadtxt('Ge_A.txt')
GeB=np.loadtxt('Ge_B.txt')


#####################################################################################
## Incertidumbre ##
def delta(M,exp):
    f=len(M)
    c=len(M[0])
    nM=np.copy(M)
    mI=np.array([[.6,6,60,600,6e3,10e3],[.1e-3,1e-3,.01,.1,.001e3,.01e3]]) #en miliAmperios
    muI=np.copy(mI)*1e3 # en microAmperios
    V=np.array([[60e-3,600e-3,6,60,600,1000],[.01e-3,.1e-3,.001,.01,.1,1]])
    if exp==0:
        I=np.copy(mI)
    elif exp==1:
        I=np.copy(muI)
    # 1ra columna amperajes
    for i in range(f):
        a=M[i,0]
        j=0
        if a<=I[0,0]:
            nM[i,j]=a*1e-2+I[1,0]*3
        elif I[0,0]<a and a<=I[0,1]:
            nM[i,j]=a*1e-2+I[1,1]*3
        elif I[0,1]<a and a<=I[0,2]:
            nM[i,j]=a*1e-2+I[1,2]*3
        elif I[0,2]<a and a<=I[0,3]:
            nM[i,j]=a*1e-2+I[1,3]*3
        elif I[0,3]<a and a<=I[0,4]:
            nM[i,j]=a*1.2e-2+I[1,4]*5
        elif a>I[0,5]:
            nM[i,j]=a*1.2e-2+I[1,5]*5
    # 2da columna voltajes//temperatura
    if exp==0:
        for i in range(f):
            a=M[i,1]
            j=1
            if a<=V[0,0]:
                nM[i,j]=a*.8e-2+V[1,0]*3            
            elif V[0,0]<a and a<=V[0,1]:
                nM[i,j]=a*.8e-2+V[1,1]*3            
            elif V[0,1]<a and a<=V[0,2]:                
                nM[i,j]=a*.5e-2+V[1,2]*1            
            elif V[0,2]<a and a<=V[0,3]:                
                nM[i,j]=a*.5e-2+V[1,3]*1            
            elif V[0,3]<a and a<=V[0,4]:                
                nM[i,j]=a*.5e-2+V[1,4]*1            
            elif a>V[0,5]:                
                nM[i,j]=a*1e-2+V[1,5]*3
    if exp==1:
        for i in range(f):
            nM[i,1]*=.05e-2
            nM[i,1]+=.3
    return np.append(M,nM,axis=1)

## Separar ensayos + incertidumbres: y,x,dy,dx

SiA1=delta(SiA[:,:2],0) # ensayo Si N 1
SiA2=delta(SiA[:,2:4],0) # ensayo Si N 2

SiB1=delta(SiB[:,:2],1) # ensayo Si N, Eg, 1
SiB2=delta(SiB[:,2:4],1) # ensayo Si N, Eg, 2

GeA1=delta(GeA[:,:2],0)
GeA2=delta(GeA[:,2:4],0)

GeB1=delta(GeB[:,:2],1)
GeB2=delta(GeB[:,2:4],1)

####################################################################################
def exp1(M):
    ee=sci.constants.e  # Coulomb
    Kb=sci.constants.k  # J.K^-1
    T=273+20
    aM=np.zeros((len(M),2))
    aM[:,0]=M[:,1]
    aM[:,1]=np.log(M[:,0])
    nM=np.zeros((len(M),1))
    for i in range(len(nM)):
        nM[i]=ee/Kb/T*np.sqrt((-M[i,2]*M[i,1]/M[i,0]/np.log(M[i,0])**2)**2 + (M[i,3]/np.log(M[i,0]))**2)
    return np.append(aM,nM,axis=1)


def exp2(M,N):
    nM=np.zeros((len(M),1))
    ee=sci.constants.e  # Coulomb
    Kb=sci.constants.k  # J.K^-1
    aM=np.zeros((len(M),2))
    aM[:,0]=(M[:,1]+273)**-1
    aM[:,1]=np.log(M[:,0])
    for i in range(len(M)):
        dN=2.53e-1
#        nM[i]=Kb/ee*np.sqrt((M[i,2]*N/M[i,0]*M[i,1])**2 + (N*M[i,3]*np.log(M[i,0]))**2 + (dN*np.log(M[i,0]*M[i,1])**2))
        nM[i]=Kb/ee*N*np.sqrt((M[i,2]/M[i,0]*M[i,1])**2 + (M[i,3]*np.log(M[i,0]))**2)
    return np.append(aM,nM,axis=1)


#####################################################################################
## Graficas ##

def cplot(M,i,exp,ax=None,**plt_kwargs):
    if ax is None:
        ax=plt.figure(i)
    qe=sci.constants.e  # Coulomb
    Kb=sci.constants.k  # J.K^-1
    Kbb=8.61733e-5
    T=273+20
    if exp==0:
        pp,err=mtd.lingen(exp1(M),mtd.pl1)
        plt.errorbar(M[:,1],M[:,0],xerr=M[:,3],yerr=M[:,2], hold=True, ecolor='k', fmt='none', label='Data')
        plt.plot(M[:,1],np.exp(pp[0]+M[:,1]*pp[1]),'--',label='$\eta=%1.4f$\n$I_0=%1.2f\;\mu A$\n$R^2=%1.2f$'%((pp[1]*Kb*T/qe)**(-1),np.exp(pp[0])*10**3,err[0]))
        plt.xlabel(r'$U\;[V]$')
        plt.ylabel('$ln\,I\;(I\;[mA])$')
        plt.semilogy(basey=np.e)
        plt.yticks([np.e**3,np.e**2,np.e,1,np.e**-1],[3,2,1,0,-1])
    if exp==1:
        pp,err=mtd.lingen(exp1(M),mtd.pl1)
        plt.errorbar(M[:,1],M[:,0],xerr=M[:,3],yerr=M[:,2], hold=True, ecolor='k', fmt='none', label='Data')
        plt.plot(M[:,1],np.exp(pp[0]+M[:,1]*pp[1]),'--',label='$\eta=%1.4f$\n$I_0=%1.2f\; mA$\n$R^2=%1.2f$'%((pp[1]*Kb*T/qe)**(-1),np.exp(pp[0]),err[0]))
        plt.xlabel(r'$U\;[V]$')
        plt.ylabel('$ln\,I\;(I\;[mA])$')
        plt.semilogy(basey=np.e)
        plt.yticks([np.e**5,np.e**4,np.e**3,np.e**2,np.e,1,np.e**-1],[5,4,3,2,1,0,-1])
    if exp==2:
        pp,err=mtd.lingen(exp2(M,1.9225),mtd.pl1)
#        plt.plot((M[:,1]+273)**(-1),M[:,0],'.-',lw=1.5,color='C1')
        plt.errorbar((M[:,1]+273)**(-1),M[:,0],yerr=M[:,2], hold=True, ecolor='k', fmt='none',label='Data')
        plt.plot((M[:,1]+273)**(-1),np.exp(pp[0]+pp[1]/(M[:,1]+273)),'--',label='$\eta=1.9225$\n $E_g=%1.2f\;eV$\n$R^2=%1.2f$'%(pp[1]*Kbb*1.9225,err[0]))
        plt.xlabel(r'$T^{-1}\;[K^{-1}]$')    
        plt.ylabel('$ln\,I\;(I\;[\mu A])$')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
        plt.semilogy(basey=np.e)
        plt.yticks([np.e**3.5,np.e**3,np.e**2.5],[3.5,3.,2.5])
    if exp==3:
        pp,err=mtd.lingen(exp2(M,1.8686),mtd.pl1)
#        plt.plot((M[:,1]+273)**(-1),M[:,0],'.-',lw=1.5,color='C1')
        plt.errorbar((M[:,1]+273)**(-1),M[:,0],yerr=M[:,2], hold=True, ecolor='k', fmt='none',label='Data')
        plt.plot((M[:,1]+273)**(-1),np.exp(pp[0]+pp[1]/(M[:,1]+273)),'--',label='$\eta=1.8686$\n $E_g=%1.2f\;eV$\n$R^2=%1.2f$'%(pp[1]*Kbb*1.8686,err[0]))
        plt.xlabel(r'$T^{-1}\;[K^{-1}]$')
        plt.ylabel('$ln\,I\;(I\;[\mu A])$')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
        plt.semilogy(basey=np.e)
        plt.yticks([np.e**3.5,np.e**3,np.e**2.5],[3.5,3.,2.5])

    plt.grid(ls=':',color='grey',alpha=.5)

    plt.legend()
#    plt.gca().add_artist(legend1)
#    plt.gca().set_ylim(bottom=0)
    return ax

cplot(SiA1,1,0)
cplot(SiA2,2,0)
cplot(GeA1,3,1)
cplot(GeA2,4,1)


cplot(SiB1,11,2)
cplot(SiB2,12,2)
cplot(GeB1,13,3)
cplot(GeB2,14,3)

plt.show()
