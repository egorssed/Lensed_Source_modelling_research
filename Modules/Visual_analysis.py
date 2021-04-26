import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import Image_Stats
import tensorflow as tf
import keras.backend as K
import ast
import seaborn as sns
import pandas as pd

image_size=64
x = np.linspace(0, 64, 64)
y = np.linspace(0, 64, 64)
X, Y = np.meshgrid(x, y)
xdata = np.vstack((X.ravel(), Y.ravel()))

def plot_galaxies(*args,dimensions='2d',show=True):
    #(num_of_rows,num_of_cols,im_shape[0],im_shape[1])
    args = [x.squeeze() for x in args]
    #num of cols
    n = min([x.shape[0] for x in args])
    I_max=np.max(args)
    print('Maximal brightness',I_max)

    if dimensions=='2d':
      fig=plt.figure(figsize=(2*n, 2*len(args)))
    else:
      fig=plt.figure(figsize=(5*n, 5*len(args)))

    for row in range(len(args)):
        for col in range(n):
            if dimensions=='2d':
                ax = fig.add_subplot(len(args),n,row*n+col+1)
                ax.imshow(args[row][col].squeeze(),cmap='Greys_r',vmax=I_max)
            else:
                ax = fig.add_subplot(len(args),n,row*n+col+1, projection='3d')
                ax.plot_surface(X, Y, args[row][col].squeeze(), cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
                ax.set_zlim(0,I_max)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.subplots_adjust(wspace=0,hspace=0)
    if show:
        plt.show()


def Show_latent_distr(models,x_test,beta_vae=1,loss_func=tf.keras.losses.binary_crossentropy):
  fig,ax=plt.subplots(1,3,figsize=(20,5))

  z_mean=K.eval(models['z_meaner'].predict(x_test))
  z_log_var=K.eval(models['z_log_varer'].predict(x_test))
  ratio=np.std(z_mean,axis=0)/np.mean(np.exp(z_log_var/2),axis=0)
  sns.barplot(ax=ax[0],x=np.linspace(0,64,64),y=ratio)
  ax[0].hlines(1,0,63,label=r'$\mu_{std}=\sigma_{mean}$')
  ax[0].legend()
  ax[0].set_xticks([])
  ax[0].set_xlabel('Latent variable')
  ax[0].set_ylabel('Ratio')
  ax[0].set_title('Latent SNR')

  #reconstruction quality
  flattened_x=K.reshape(x_test,shape=(len(x_test),image_size*image_size))
  flattened_decoded=K.reshape(models['vae'].predict(x_test),shape=(len(x_test),image_size*image_size))
  Log_loss=image_size*image_size*loss_func(flattened_x,flattened_decoded)

  KL_loss=0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

  df=pd.DataFrame()
  df['Regularization loss']=beta_vae*KL_loss
  df['Reconstruction loss']=Log_loss
  sns.barplot(ax=ax[1],data=df[['Reconstruction loss']])
  sns.barplot(ax=ax[2],data=df[['Regularization loss']])
  plt.tight_layout()
  plt.show()

def learning_curve(filename,start_epoch=0,stop_epoch=1000):
  logs_file=open(filename)
  lines=logs_file.readlines()
  logs_file.close()


  loss=np.array([])
  val_loss=np.array([])
  for line in lines:
    note=ast.literal_eval(line)
    loss=np.append(loss,[note['loss']])
    val_loss=np.append(val_loss,[note['val_loss']])

  start_index=start_epoch//10
  stop_index=np.minimum(len(loss),stop_epoch//10+1)
  plt.plot(10*np.arange(start_index,stop_index),loss[start_index:stop_index],label='Train')
  plt.plot(10*np.arange(start_index,stop_index),val_loss[start_index:stop_index],label='Validation')
  plt.ylabel('Loss')
  plt.xlabel('epoch number')
  plt.title('Learning curve')
  plt.legend()
  plt.show()

def present_reconstruction(models,imgs,dimensions='2d',resid=False):
    #Images selection
    images_for_reconst=imgs
    decoded_to_reconstruct=models['vae'].predict(images_for_reconst, batch_size=len(imgs))
    if resid:
      residuals=decoded_to_reconstruct-images_for_reconst
      plot_galaxies(images_for_reconst[:10],decoded_to_reconstruct[:10],residuals[:10],dimensions=dimensions,show=True)
    else:
      plot_galaxies(images_for_reconst[:10],decoded_to_reconstruct[:10],dimensions=dimensions,show=False)
      plt.colorbar()
      plt.show()


