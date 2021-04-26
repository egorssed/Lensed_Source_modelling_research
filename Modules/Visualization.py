import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import Image_Stats
from scipy.special import gammaincinv
import copy

def present_3D(image,Sersic_fit=None):
    x = np.linspace(0, image.shape[1], image.shape[1])
    y = np.linspace(0, image.shape[0], image.shape[0])
    X, Y = np.meshgrid(x, y)
    # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
    xdata = np.vstack((X.ravel(), Y.ravel()))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, image, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Image surface brightness')

    if (Sersic_fit is not None) and len(Sersic_fit)==7:
        image_pred=Image_Stats.Sersic_profile(xdata, *Sersic_fit).reshape(image.shape)

        fig_fit, ax_fit = plt.subplots(subplot_kw={"projection": "3d"})
        surf_fit = ax_fit.plot_surface(X, Y, image_pred, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

        ax_fit.set_xticks([])
        ax_fit.set_yticks([])
        ax_fit.set_title('Prediction surface brightness')

        _,ax=plt.subplots(1,2)
        ax[0].imshow(image,cmap='gray')
        ax[0].set_title('Image')
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].imshow(image_pred,cmap='gray')
        ax[1].set_title('Prediction')
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        plt.tight_layout()

    else:
        plt.figure(3)
        plt.imshow(image,cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('Image')
    plt.show()

def present_Image_cut(image,R,Radial_profile=None,Radial_profile_threshold=0,
                       Increase_curve=None,Increase_flux_threshold=1,Sersic_fit=None,Sersic_flux_threshold=1):

    Stats=[]
    R_cuts=[]
    Titles=[]

    if Radial_profile is not None:
        Radial_profile=Radial_profile-Radial_profile[-1]
        Radial_profile/=Radial_profile.max()
        R_cut=np.where((Radial_profile<Radial_profile_threshold))[0]
        if len(R_cut)==0:
            R_cut=np.inf
        else:
            R_cut=R_cut[0]

        R_cuts.append(R_cut)
        Stats.append(Radial_profile)
        Titles.append(('Radial profile','RP>{}'.format(Radial_profile_threshold)))

    if Increase_curve is not None:
        Increase_curve/=Increase_curve.max()
        R_cut=np.where((Increase_curve>Increase_flux_threshold))[0]
        if len(R_cut)==0:
            R_cut=np.inf
        else:
            R_cut=R_cut[0]

        R_cuts.append(R_cut)
        Stats.append(Increase_curve)
        Titles.append(('Increase curve','{}% of total flux'.format(Increase_flux_threshold*100)))

    if Sersic_fit is not None:
        HLR,n=Sersic_fit[1:3]
        k=2*n-0.324
        if Sersic_flux_threshold!=1:
            R_cut=np.power(gammaincinv(2*n,Sersic_flux_threshold)/k,n)*HLR
        else:
            R_cut=np.inf

        R_cuts.append(R_cut)
        Titles.append(('Sersic profile','{}% of total Sersic flux'.format(Sersic_flux_threshold*100)))


    Number_of_plots=1+len(R_cuts)+(len(Stats)!=0)
    fig,ax=plt.subplots(1,Number_of_plots,figsize=(20,5))
    ax[0].imshow(image,cmap='gray')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Image')

    for i,R_cut in enumerate(R_cuts):
        image_with_cut=copy.deepcopy(image)
        image_with_cut[(R>R_cut)]=image_with_cut.max()
        ax[i+1].imshow(image_with_cut,cmap='gray')
        ax[i+1].set_xticks([])
        ax[i+1].set_yticks([])
        ax[i+1].set_title(Titles[i][0])
        ax[i+1].set_xlabel(Titles[i][1])


    for i,Stat in enumerate(Stats):
        ax[-1].plot(Stat,label=Titles[i][0],color='C{}'.format(i))
        ax[-1].vlines(R_cuts[i],0,1,label=Titles[i][1],colors='C{}'.format(i))
        ax[-1].legend()

    plt.tight_layout()
    plt.show()

def present_Image_cut_2(image,R,Radial_profile=None,Radial_profile_threshold=0,
                       Increase_curve=None,Increase_flux_threshold=1,Sersic_fit=None,Sersic_flux_threshold=1):

    Stats=[]
    R_cuts=[]
    Titles=[]

    if Radial_profile is not None:
        #Radial_profile=Radial_profile-Radial_profile[-1]
        Radial_profile/=np.nanmax(Radial_profile)
        Radial_profile-=Radial_profile[-1]
        R_cut=np.where((Radial_profile<Radial_profile_threshold))[0]
        if len(R_cut)==0:
            R_cut=np.inf
        else:
            R_cut=R_cut[0]

        R_cuts.append(R_cut)
        Stats.append(Radial_profile)
        Titles.append(('Radial profile','RP>{}'.format(Radial_profile_threshold)))

    if Increase_curve is not None:
        Increase_curve/=Increase_curve.max()
        R_cut=np.where((Increase_curve>Increase_flux_threshold))[0]
        if len(R_cut)==0:
            R_cut=np.inf
        else:
            R_cut=R_cut[0]

        R_cuts.append(R_cut)
        Stats.append(Increase_curve)
        Titles.append(('Increase curve','{}% of total flux'.format(Increase_flux_threshold*100)))

    if Sersic_fit is not None:
        HLR,n=Sersic_fit[1:3]
        k=2*n-0.324
        if Sersic_flux_threshold!=1:
            R_cut=np.power(gammaincinv(2*n,Sersic_flux_threshold)/k,n)*HLR
        else:
            R_cut=np.inf

        R_cuts.append(R_cut)
        Titles.append(('Sersic profile','{}% of total Sersic flux'.format(Sersic_flux_threshold*100)))


    #fig=plt.figure(figsize=(20,7))

    ax0=plt.subplot2grid((2,4),(0,0))
    ax0.imshow(image,cmap='gray')
    ax0.set_xticks([])
    ax0.set_yticks([0,len(image)//2,len(image)])
    ax0.set_yticklabels([len(image),len(image)//2,0])
    ax0.set_title('Image')

    for i,R_cut in enumerate(R_cuts):
        image_with_cut=copy.deepcopy(image)
        image_with_cut[(R>R_cut)]=image_with_cut.max()
        ax=plt.subplot2grid((2,4),(0,i+1))
        ax.imshow(image_with_cut,cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(Titles[i][0])
        ax.set_xlabel(Titles[i][1])

    ax=plt.subplot2grid((2,4),(1,0),colspan=4)
    for i,Stat in enumerate(Stats):
        ax.plot(Stat,label=Titles[i][0],color='C{}'.format(i))
        ax.vlines(R_cuts[i],0,1,label=Titles[i][1],colors='C{}'.format(i))
        ax.set_xlabel('Radius')

    ax.legend()
    plt.tight_layout()
