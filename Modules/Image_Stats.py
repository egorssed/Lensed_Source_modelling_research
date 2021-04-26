import numpy as np
from scipy.optimize import curve_fit
import copy

def Sersic_profile(M,I,HLR,n,q,x0,y0,phi):
        (x, y) = M
        #See Lackner & Gunn 2012 for details on Sersic profile
        R=np.sqrt(np.power((x-x0)*np.cos(phi)+(y-y0)*np.sin(phi),2)+
                np.power((y-y0)*np.cos(phi)-(x-x0)*np.sin(phi),2)/np.power(q,2))
        #Ciotti 1991
        k=2*n-0.324
        S=I*np.exp(-k*(np.power(R/HLR,1/n)-1))
        return S

def Chameleon_profile(M,I,Wt,Wratio,q,x0,y0,phi):
  (x, y) = M
  Wc=Wt*Wratio
  Rsq=np.power((x-x0)*np.cos(phi)+(y-y0)*np.sin(phi),2)+np.power((y-y0)*np.cos(phi)-(x-x0)*np.sin(phi),2)/np.power(q,2)

  def Isothermal_profile(w):
    Softening=np.power(2*w/(1+q),2)
    return np.power(Rsq+Softening,-0.5)

  return I/(1+q)*(Isothermal_profile(Wc)-Isothermal_profile(Wt))

def Chameleon_Sersic(M,I1,Wt1,Wratio1,q1,phi1,I2,HLR,n,q2,phi2,x0,y0):
    return Chameleon_profile(M,I1,Wt1,Wratio1,q1,x0,y0,phi1)+Sersic_profile(M,I2,HLR,n,q2,x0,y0,phi2)

def Double_Sersic(M,I1,HLR1,n1,q1,phi1,I12ratio,HLR2,n2,q2,phi2,x0,y0):
    I2=I1*I12ratio
    return Sersic_profile(M,I1,HLR1,n1,q1,x0,y0,phi1)+Sersic_profile(M,I2,HLR2,n2,q2,x0,y0,phi2)



def fit_image(image,profile_type='Sersic'):
    x = np.linspace(0, image.shape[1], image.shape[1])
    y = np.linspace(0, image.shape[0], image.shape[0])
    X, Y = np.meshgrid(x, y)
    # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
    xdata = np.vstack((X.ravel(), Y.ravel()))

    if profile_type=='Sersic':
        func=Sersic_profile
        n_initial=1
        k=2*n_initial-0.324
        initial_guess=np.array([image.max()*np.exp(-k),5,n_initial,0.64,
                            image.shape[1]/2,image.shape[0]/2,0])
        lower_bounds=np.array([image.max()*1e-5,0,0.5,0.01,0,0,-np.pi/2])
        upper_bounds=np.array([image.max(),np.min(image.shape)/2,6,1,
                           image.shape[1],image.shape[0],np.pi/2])

    elif profile_type=='Chameleon':
        func=Chameleon_profile
        initial_guess=np.array([image.max(),5,0.5,0.5,image.shape[1]/2,image.shape[0]/2,0])
        lower_bounds=np.array([image.max()*1e-5,0,0,0.01,0,0,-np.pi/2])
        upper_bounds=np.array([np.inf,np.inf,1,1,image.shape[1],image.shape[0],np.pi/2])

    elif profile_type=='Chameleon_Sersic':
        func=Chameleon_Sersic
        n_initial=1
        k=2*n_initial-0.324
        initial_guess=np.array([image.max()*20,2,0.1,0.64,0,
                             image.max()*np.exp(-k),5,n_initial,0.64,0,
                             image.shape[1]/2,image.shape[0]/2])
        lower_bounds=np.array([image.max()*1e-5,0,0,0.01,-np.pi/2,
                           0,0,0.5,0.01,-np.pi/2,
                           0,0])
        upper_bounds=np.array([np.inf,np.inf,1,1,np.pi/2,
                           image.max(),np.min(image.shape)/2,6,1,np.pi/2,
                           image.shape[1],image.shape[0]])
    elif profile_type=='Double_Sersic':
        func=Double_Sersic
        n_initial=np.array([4,1])
        k=2*n_initial-0.324
        Iratio_init=np.exp(-k[0]+k[1])
        initial_guess=np.array([image.max()*np.exp(-k[0]),2,n_initial[0],0.64,0,
                             Iratio_init,5,n_initial[1],0.64,0,
                             image.shape[1]/2,image.shape[0]/2])
        lower_bounds=np.array([image.max()*1e-5,0,0.5,0.01,-np.pi/2,
                           0,0,0.5,0.01,-np.pi/2,
                           0,0])
        upper_bounds=np.array([image.max(),np.min(image.shape)/2,6,1,np.pi/2,
                           1,np.min(image.shape)/2,6,1,np.pi/2,
                           image.shape[1],image.shape[0]])

    else:
        print('Wrong profile type')
        return False

    #Poisson errors
    sigma=np.sqrt(np.abs(image)+image.max()*1e-5)

    popt, pcov = curve_fit(f=func, xdata=xdata,
                               ydata=image.ravel(),
                               p0=initial_guess,
                               sigma=sigma.ravel(),
                               bounds=(lower_bounds,upper_bounds))
    return popt,np.sqrt(np.diag(pcov))

def chi_sq(image_true,image_pred):

    x = np.linspace(0, image_true.shape[1], image_true.shape[1])
    y = np.linspace(0, image_true.shape[0], image_true.shape[0])
    X, Y = np.meshgrid(x, y)
    # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
    xdata = np.vstack((X.ravel(), Y.ravel()))

    true=image_true.ravel()
    pred=image_pred.ravel()
    sigma=np.sqrt(np.abs(true)+true.max()*1e-5)

    return (((true-pred)/sigma)**2).sum()

def Radial_profile(image,q=None,x0=None,y0=None,phi=None,Increase_curve=False):
    if x0 is None or y0 is None:
        x0=image.shape[1]/2
        y0=image.shape[0]/2
    if q is None or phi is None:
        q=1
        phi=0
    R_max=np.min(image.shape)//2
    radial_profile=np.zeros(R_max)
    counter=np.zeros(R_max)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            R=np.sqrt(np.power((x-x0)*np.cos(phi)+(y-y0)*np.sin(phi),2)+
                np.power((y-y0)*np.cos(phi)-(x-x0)*np.sin(phi),2)/np.power(q,2))
            if(R<R_max):
                radial_profile[int(R)]+=image[y,x]
                counter[int(R)]+=1

    if Increase_curve:
        increase_curve=copy.deepcopy(radial_profile)
        for i in range(1,R_max):
            increase_curve[i]+=increase_curve[i-1]

    mask_nan=(counter==0)
    counter[mask_nan]=np.nan
    radial_profile[mask_nan]=np.nan
    radial_profile=radial_profile/counter

    #Radial profile is average Flux in a ring of radius R
    #Increase curve is a total Flux in a circle of radius R
    if Increase_curve:
        return radial_profile,increase_curve
    else:
        return radial_profile

def compare_fits(image_true,image_pred,profile_type='Chameleon_Sersic'):
    true_fit,_=fit_image(image_true,profile_type)
    pred_fit,_=fit_image(image_pred,profile_type)
    RAE=np.abs((true_fit-pred_fit)/true_fit)
    return true_fit,pred_fit,RAE
