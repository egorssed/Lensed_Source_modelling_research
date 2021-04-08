import numpy as np
import pandas as pd
import galsim
import Image_Stats
from tqdm import tqdm
import sys

#Size of region to extract the noise from
noise_border_size=8

#Galsim arguments
target_size=64
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


def Manual_noise_extraction(image):
    #Border regions for noise extraction
    Borders=np.array([image[:noise_border_size,:noise_border_size],image[:noise_border_size,-noise_border_size:],
                    image[-noise_border_size:,:noise_border_size],image[-noise_border_size:,-noise_border_size:]])

    #Taking min to make sure that we take std of only background
    #It enables us to avoid accounting std for noise+border of
    #diagonal huge edge on spiral galaxies touching noise extraction regions
    noise_mean=min(Borders.mean(axis=(1,2)))
    noise_std=min(Borders.std(axis=(1,2)))
    return noise_mean,noise_std

def create_Galaxy(index,Radial_profile_threshold=0.01):
    #Making galaxy
    gal=cat.makeGalaxy(index,gal_type=galaxy_type)
    #Get the original HST image
    gal=Convolve_with_PSF(gal)
    original_image=gal.drawImage(use_true_center=True, method='auto').array

    try:
        #Extracting the parameters to build elliptical coordinates
        Sersic_fit,_=Image_Stats.fit_image(original_image)
        q,x0,y0,phi=Sersic_fit[-4:]
        Radial_profile=Image_Stats.Radial_profile(original_image,q,x0,y0,phi)

        #Getting border of galaxy from radial profile
        Radial_profile=Radial_profile-Radial_profile[-1]
        Radial_profile/=Radial_profile.max()
        R_cut=np.where((Radial_profile<Radial_profile_threshold))[0]
        if len(R_cut)==0:
            R_cut=np.inf
        else:
            R_cut=R_cut[0]
    except:
        print('Unsuccesfull original image processing')
        return False


    if np.min(original_image.shape)<target_size:
        #Check that the original image is bigger than the desired one
        #We for sure don't want to upscale anything, only downscale if needed
        print('Original size is less than the target size')
        return False
    elif np.min(original_image.shape)<2*R_cut:
        #If the image encloses less than given threshold of luminosity
        #Than there is no sense to perform the cut. Downscale the image to the target 64x64
        #The scale may seem strange but Galsim usually works in terms Galsim_size=Image_size-2
        Galsim_scale=pixel_scale*(np.min(original_image.shape)-2)/(target_size-2)
        image_64x64= gal.drawImage(scale=Galsim_scale,use_true_center=True, method='auto').array

        #Sanity check cause Galsim sometimes do weird stuff
        if image_64x64.shape!=(64,64):
            print('Wrong galsim shape, index:',index)
            return False
    else:
        #So here we want to crop the image and downscale it in such a way that 2*R_cut->64
        #But we don't want to upscale anything, so put a constraint of 1
        Scaling_factor=np.minimum(target_size/(2*R_cut),1)
        if Scaling_factor!=1:
            Galsim_scale=pixel_scale*(2*R_cut-2)/(target_size-2)
            image_scaled=gal.drawImage(scale=Galsim_scale,use_true_center=True, method='auto').array
        else:
            image_scaled=original_image

        if image_scaled.shape<(64,64):
            print('Wrong galsim shape, index:',index)
            return False

        #Perform the crop
        x0=image_scaled.shape[1]//2
        y0=image_scaled.shape[0]//2
        image_64x64=image_scaled[y0-32:y0+32,x0-32:x0+32]

    #Create description of galaxy features
    parameters=pd.Series(cat.getParametricRecord(index))[['IDENT', 'mag_auto',  'zphot']]
    parameters['COSMOS_use_bulgefit']= cat.getParametricRecord(index)['use_bulgefit']
    #Extract the fits of COSMOS dataset and drop boxiness
    COSMOS_Sersic=np.delete(pd.Series(cat.getParametricRecord(index))['sersicfit'],4)
    COSMOS_Sersic=pd.Series(data=COSMOS_Sersic,index=['COSMOS_Sersic_I','COSMOS_Sersic_HLR','COSMOS_Sersic_n','COSMOS_Sersic_q',
                                                      'COSMOS_Sersic_x0','COSMOS_Sersic_y0','COSMOS_Sersic_phi'])

    COSMOS_Bulge_Disk=np.delete(pd.Series(cat.getParametricRecord(index))['bulgefit'],[4,12])
    COSMOS_Disk=pd.Series(data=COSMOS_Bulge_Disk[:7],
                            index=['COSMOS_Disk_I','COSMOS_Disk_HLR','COSMOS_Disk_n','COSMOS_Disk_q',
                'COSMOS_Disk_x0','COSMOS_Disk_y0','COSMOS_Disk_phi'])
    COSMOS_Bulge=pd.Series(data=COSMOS_Bulge_Disk[7:],
                            index=['COSMOS_Bulge_I','COSMOS_Bulge_HLR','COSMOS_Bulge_n','COSMOS_Bulge_q',
                'COSMOS_Bulge_x0','COSMOS_Bulge_y0','COSMOS_Bulge_phi'])

    _,_,_,_,COSMOS_noise=cat.getRealParams(index)
    parameters['COSMOS_noise']=COSMOS_noise
    parameters=parameters.append(COSMOS_Sersic)
    parameters=parameters.append(COSMOS_Bulge)
    parameters=parameters.append(COSMOS_Disk)

    #Fit the final image with Sersic
    try:
        Sersic_fit,Sersic_errors=Image_Stats.fit_image(image_64x64)
    except:
        Sersic_fit=np.full(7,np.nan)
        Sersic_errors=np.full(7,np.nan)

    Sersic_vals=pd.Series(data=Sersic_fit,index=['Sersic_I','Sersic_HLR','Sersic_n','Sersic_q',
                'Sersic_x0','Sersic_y0','Sersic_phi'])
    Sersic_errs=pd.Series(data=Sersic_errors,index=['Sersic_I_err','Sersic_HLR_err','Sersic_n_err','Sersic_q_err',
                'Sersic_x0_err','Sersic_y0_err','Sersic_phi_err'])

    parameters=parameters.append(Sersic_vals)
    parameters=parameters.append(Sersic_errs)

    parameters.name=index

    noise_mean,noise_std=Manual_noise_extraction(image_64x64)
    parameters['Noise_mean']= noise_mean
    parameters['Noise_std']= noise_std


    parameters['max_I']= image_64x64.max()
    parameters['min_I']= image_64x64.min()
    parameters['Original_x_size']=original_image.shape[1]
    parameters['Original_y_size']=original_image.shape[0]
    parameters['Original_R_cut']=R_cut
    parameters['Sersic_r_chi_sq']=Image_Stats.chi_sq(image_64x64,Sersic_fit)


    return image_64x64,parameters

def main(start,stop):

    images=np.zeros((0,64,64))
    labels=pd.DataFrame()

    for index in tqdm(range(start,np.minimum(stop,cat.nobjects))):
        try:
            result=create_Galaxy(index,0.01)
        except:
            continue

        if result==False:
            print('index:',index)
            continue
        else:
            image,parameters=result

        #Push results to storages
        labels=labels.append(parameters)
        images=np.append(images,[image],axis=0)

    labels.to_csv('Data/Labels_{start}_{stop}.csv'.format(start=start,stop=stop))
    np.save('Data/Images_{start}_{stop}.npy'.format(start=start,stop=stop),images)


def is_intstring(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

if __name__ == '__main__':

    for arg in sys.argv[1:]:
        if not is_intstring(arg):
            sys.exit("All arguments must be integers. Exit.")

    arguments=[int(arg) for arg in sys.argv[1:]]
    if len(arguments)==2:
        start=arguments[0]
        stop=arguments[1]
        main(start,stop)
    else:
        print('Wrong number of arguments. Please, insert "start stop" indices')

