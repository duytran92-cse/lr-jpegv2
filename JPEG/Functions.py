#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 15:02:27 2018

@author: rpeteri
"""
# Fonction codage DCT/JPEG
import numpy  as np

mNORM= np.ones([8,8])
mNORM[1:8,0]=1/np.sqrt(2)   #np.array([[1/2*  np.ones((1,7))/np.sqrt(2)],[np.ones((7,1))/np.sqrt(2)],[np.ones((7,7))] ])/4
mNORM[0,1:8]=1/np.sqrt(2)
mNORM[0,0]=0.5
mNORM=mNORM/4

def DCT(pixc):
    """Calcul les coefficients DCT d'un bloc 8*8 
        :param pixc: (8*8) bloc 
        :param coeff:DCT coefficients (8*8)   
        """

    #pixc=float(pixc);
    global mNORM
    tmp=np.repeat(np.arange(0,8)[:,None],1,axis=1) #column vector
    mUX=np.cos(tmp*(2*np.arange(0,8)+1)*np.pi/16)
    mYV=np.cos((2*tmp+1)*np.arange(0,8)*np.pi/16)
    coeff=mNORM*(mUX@pixc@mYV)
    return coeff


def iDCT(Pdct):
    """Calcul l'inverse DCT d'un bloc 8*8 
        :param Pdct:DCT coefficients (8*8)
        :param pixels:(8*8) bloc 
         """
    global mNORM
    tmp=np.repeat(np.arange(0,8)[:,None],1,axis=1) #column vector
    mUX=np.cos(tmp*(2*np.arange(0,8)+1)*np.pi/16)
    mYV=np.cos((2*tmp+1)*np.arange(0,8)*np.pi/16)
    coeff=mYV@(mNORM*Pdct)@mUX
    return coeff

