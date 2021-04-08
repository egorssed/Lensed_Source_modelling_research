import numpy as np
import galsim
import matplotlib.pyplot as plt
import Image_Stats
import pandas as pd
import Visualization

#Size of region to extract the noise from
noise_border_size=8

#Galsim arguments
target_size=64
Luminosity_threshold=1-(1e-3)
galaxy_type='real'
psf_type='real'

cat = galsim.COSMOSCatalog(sample='23.5')
#Pixel scale in arcsec (the same for all the galaxies)
_,_,_,pixel_scale,_=cat.getRealParams(0)

def Convolve_with_PSF(gal):
    #Get the PSF object
    psf = gal.original_psf
    #Perform convolution with PSF
    return galsim.Convolve(gal, psf)

def get_image(index):
    gal=cat.makeGalaxy(index,gal_type=galaxy_type)
    gal=Convolve_with_PSF(gal)
    return gal.drawImage(use_true_center=True, method='auto').array

def get_R(image,Sersic_fit):
    q,x0,y0,phi=Sersic_fit[-4:]
    x = np.linspace(0, image.shape[1], image.shape[1])
    y = np.linspace(0, image.shape[0], image.shape[0])
    X, Y = np.meshgrid(x, y)

    R=np.sqrt(np.power((X-x0)*np.cos(phi)+(Y-y0)*np.sin(phi),2)+
                np.power((Y-y0)*np.cos(phi)-(X-x0)*np.sin(phi),2)/np.power(q,2))

    return R

if __name__ == '__main__':
    indices=[19900,28193,28419,20055,17269]
    index=37

    image=get_image(index)
    Sersic_fit,_=Image_Stats.fit_image(image)
    print(Sersic_fit)
    print(Image_Stats.chi_sq(image, Sersic_fit))
    #Visualization.present_3D(image,Sersic_fit)


    q,x0,y0,phi=Sersic_fit[-4:]
    Radial_profile,Increase_curve=Image_Stats.Radial_profile(image,q,x0,y0,phi,Increase_curve=True)

    Visualization.present_Image_cut_2(image,get_R(image,Sersic_fit),Radial_profile,0.01,Increase_curve,0.95,Sersic_fit,0.95)
    #plt.savefig('Images for cut methods/galaxy_{}'.format(index))
    plt.show()


