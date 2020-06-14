# -*- coding: utf-8 -*-
"""
    Created on Thu Mar 24 23:08:22 2016
    
    @author: rpeteri
    """

import numpy  as np
import scipy.ndimage as ndsig

def decompose(filtre_ligne, filtre_colonne,im_in):
    """Decompose une image par un filtre sur les lignes epuis un filtre sur les colonnes
        :param filtre_ligne:
        :param filtre_colonne:
        :param im_in:
        """
    im_out=np.double(np.zeros(im_in.shape))
    print(im_out.shape)
    print("Decomposition function!")
    tmp= ndsig.filters.convolve1d(np.double(im_in),filtre_ligne,axis=0,mode='constant') # equivalent à conv2(filtre_ligne,im_in,'same') en Matlab
    im_out= ndsig.filters.convolve1d(tmp,filtre_colonne,axis=1,mode='constant') # equivalent à conv2(filtre_colonne,im_in,'same') en Matlab
    im_out=im_out[::2,::2]
    return im_out

def reconstruction(filtre_ligne, filtre_colonne,im_in):
    a_ins_nbr, a_ins_nbc = tuple(int(i * 2) for i in im_in.shape)
    a_ins=np.double(np.zeros([a_ins_nbr, a_ins_nbc]))
    a_ins[1::2,1::2]=im_in;
    tmp= ndsig.filters.convolve1d(np.double(a_ins),filtre_ligne,axis=0,mode='constant') # equivalent à conv2(filtre_ligne,im_in,'same') en Matlab
    im_out= ndsig.filters.convolve1d(tmp,filtre_colonne,axis=1,mode='constant') # equivalent à conv2(filtre_colonne,im_in,'same') en Matlab
    print(im_out.shape)
    print("Reconstruction function!")
    return im_out


def composite_image(im1,im2,im3,im4):
    (rows,cols) = im1.shape
    im_out = np.zeros((2*rows, 2*cols), dtype=np.double)
    im_out[:rows, :cols] = im1
    im_out[:rows, cols:] = im2
    im_out[rows:, :cols] = im3
    im_out[rows:, cols:] = im4
    print("Create composite image!")
    return im_out

