"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2022 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker

def create_polar_plot(ax,img,cmap,norm,elev=0.5,az1=None,az2=None,
                      r1=None,r2=None, contours=None,annotate=True):
    rvec=np.linspace(2125, 2125 + 250 * img.shape[1]-1, img.shape[1])
    azmax=360 + 360/img.shape[0]
    azvec=np.pi/180*np.linspace(0,azmax,img.shape[0]+1)
    R,AZ=np.meshgrid(rvec,azvec)
    Z=np.concatenate((img,img[0:1,:]))
    elev = elev*np.pi/180
    X,Y=R*np.sin(AZ)*np.cos(elev),R*np.cos(AZ)*np.cos(elev)
    im=ax.pcolormesh(X,Y,Z,cmap=cmap,norm=norm,shading='nearest')
    
    if contours is not None:
        ax.contour(X,Y,Z ,contours, colors='k',interpolation='none',linewidths=.25,alpha=0.5)
    
    if az1:
        x1=r1*np.sin(az1*np.pi/180)*np.cos(elev)
        x2=r2*np.sin(az2*np.pi/180)*np.cos(elev)
        y1=r1*np.cos(az1*np.pi/180)*np.cos(elev)
        y2=r2*np.cos(az2*np.pi/180)*np.cos(elev)
        ax.plot([x1,x2],[y1,y2],'k.-',linewidth=3)
    
    #plot radial and azmuthal lines
    rm,azm=np.meshgrid(np.arange(0,350e3,50e3),
                       np.arange(0,360,20))
    Xm,Ym=rm*np.sin(azm*np.pi/180),rm*np.cos(azm*np.pi/180)

    for i in range(Xm.shape[0]):
        ax.plot(Xm[i,:],Ym[i,:],'-',color=[.5,.5,.5],alpha=.5,linewidth=.5)
        if annotate:
            ax.text(1.01*Xm[i,-1],1.01*Ym[i,-1],str(azm[i,0]),color=[.5,.5,.5],backgroundcolor='w',fontsize=3)
    for r in np.arange(0,350e3,50e3):
        circ=plt.Circle((0, 0), r, linestyle='-',linewidth=.5, edgecolor=[.5,.5,.5],facecolor=None,fill=None)
        ax.add_patch(circ)
        if r>0 and r<300e3:
            if annotate:
                ax.text(r,0,str(int(r/1e3)),color=[.5,.5,.5],fontsize=3)#,backgroundcolor=[.9,.9,.9]
    
    return im,ax

def add_cbar(fig,ax,im,label='m/s'):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar=fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label(label)
    return cbar
