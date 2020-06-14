import skimage  as sci

import JP2000.Functions as jp2

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndsig
import numpy.linalg as nl

# img = np.double(plt.imread('./cameraman.tif'))

LO_D=[1/np.sqrt(2),1/np.sqrt(2)]
HI_D=[-1/np.sqrt(2),1/np.sqrt(2)]

def decomposer(img, LO_D, HI_D, compteur):
    compteur -= 1
    if(compteur == 0):
        a1 = jp2.decompose(LO_D, LO_D, img)
    else:
        a1 = decomposer(jp2.decompose(LO_D, LO_D, img), LO_D, HI_D, compteur)
    im_out=np.double(np.zeros(img.shape))
    d1 = jp2.decompose(HI_D, HI_D, img)
    h1 = jp2.decompose(LO_D, HI_D, img)
    v1= jp2.decompose(HI_D, LO_D, img)
    
    im_out[0:a1.shape[0],0:a1.shape[1]] = a1
    im_out[0:a1.shape[0],a1.shape[0]:2*a1.shape[0]] = h1
    im_out[a1.shape[0]:2*a1.shape[0],0:a1.shape[1]] = v1
    im_out[a1.shape[0]:2*a1.shape[0],a1.shape[0]:2*a1.shape[0]] = d1
    return im_out

# cpt = 2
# imgCompressee = decomposer(img, LO_D, HI_D, cpt)

LO_R=[1/np.sqrt(2),1/np.sqrt(2)]
HI_R=[1/np.sqrt(2),-1/np.sqrt(2)]

def recomposer(img, LO_R, HI_R, compteur):
    compteur -= 1
    sizeX = int(img.shape[0]/2)
    sizeY = int(img.shape[1]/2)
    a1 = img[0:sizeX,0:sizeY]
    h1 = img[0:sizeX,sizeY:2*sizeY]
    v1 = img[sizeX:2*sizeX,0:sizeY]
    d1 = img[sizeX:2*sizeX,sizeY:2*sizeY]
    if(compteur == 0):
        a1_r = jp2.reconstruction(LO_R, LO_R,a1)
    else:
        a1_r = jp2.reconstruction(LO_R, LO_R,recomposer(a1, LO_R, HI_R, compteur))
    h1_r = jp2.reconstruction(LO_R, HI_R,h1)
    v1_r = jp2.reconstruction(HI_R, LO_R,v1)
    d1_r = jp2.reconstruction(HI_R, HI_R,d1)
    img = a1_r + h1_r + v1_r + d1_r
    return img

def decomposerSeuil(img, LO_D, HI_D, compteur, seuil):
    compteur -= 1
    if(compteur == 0):
        a1 = jp2.decompose(LO_D, LO_D, img)
    else:
        a1 = decomposerSeuil(jp2.decompose(LO_D, LO_D, img), LO_D, HI_D, compteur, seuil)
    im_out=np.double(np.zeros(img.shape))
    d1 = jp2.decompose(HI_D, HI_D, img)
    h1 = jp2.decompose(LO_D, HI_D, img)
    v1= jp2.decompose(HI_D, LO_D, img)
    #pour seuiller les ondelettes: :
    h1 = np.where(np.abs(h1) > seuil, h1, 0.0)
    v1 = np.where(np.abs(v1) > seuil, v1, 0.0)
    d1 = np.where(np.abs(d1) > seuil, d1, 0.0)
    
    im_out[0:a1.shape[0],0:a1.shape[1]] = a1
    im_out[0:a1.shape[0],a1.shape[0]:2*a1.shape[0]] = h1
    im_out[a1.shape[0]:2*a1.shape[0],0:a1.shape[1]] = v1
    im_out[a1.shape[0]:2*a1.shape[0],a1.shape[0]:2*a1.shape[0]] = d1
    return im_out

def applyContourByBias(image, discretZone, deathZone):
    contour = np.array(image[int(image.shape[0]/3):2*int(image.shape[0]/3),int(image.shape[0]/3):2*int(image.shape[0]/3)])
    for rangX in range(len(image)):
        for rangY in range(len(image[0])):
            if(image[rangX][rangY] != 0.0):
                image[rangX][rangY] = int(image[rangX][rangY]/discretZone)*discretZone
    image = np.where(np.abs(image) > deathZone, image, 0.0)
    image[int(image.shape[0]/3):2*int(image.shape[0]/3),int(image.shape[0]/3):2*int(image.shape[0]/3)] = contour
    return image # return an image applied by contour with bias

def biasDecomposing(img, LO_D, HI_D, compteur, deathZone, discretZone, size, energie):
    compteur -= 1
    if(compteur == 0):
        a1 = jp2.decompose(LO_D, LO_D, img)
    else:
        a1, size, energie = biasDecomposing(jp2.decompose(LO_D, LO_D, img), LO_D, HI_D, compteur, deathZone, discretZone, size, energie)
        size += size
        energie += energie
    im_out=np.double(np.zeros(img.shape))
    d1 = jp2.decompose(HI_D, HI_D, img)			# similar to decompose() -> find depth
    h1 = jp2.decompose(LO_D, HI_D, img)			# similar to decompose() -> find horizontal
    v1= jp2.decompose(HI_D, LO_D, img)			# similar to decompose() -> find vertical

    h1 = applyContourByBias(h1, discretZone, deathZone)		# apply contour by bias on h1 to find horizontal
    v1 = applyContourByBias(v1, discretZone, deathZone)		# apply contour by bias on v1 to find vertical
    d1 = applyContourByBias(d1, discretZone, deathZone)		# apply contour by bias on d1 to find depth
    
    size += (h1.size - np.count_nonzero(h1) + v1.size - np.count_nonzero(v1) + d1.size - np.count_nonzero(d1))
    energie += ((h1**2).sum() + (v1**2).sum() + (d1**2).sum())
    
    im_out[0:a1.shape[0],0:a1.shape[1]] = a1
    im_out[0:a1.shape[0],a1.shape[0]:2*a1.shape[0]] = h1
    im_out[a1.shape[0]:2*a1.shape[0],0:a1.shape[1]] = v1
    im_out[a1.shape[0]:2*a1.shape[0],a1.shape[0]:2*a1.shape[0]] = d1
    return im_out, size, energie # im_out is a string -> but you can store the returned value by array -> return [a1, h1, v1, d1]
    # return energy to plot

##### decompose Image with Wavelet -> load input image + decompose #####

def decomposeImageWithWavelet(cpt, cheminImage, deathZone, discretZone):
    img = np.double(plt.imread(cheminImage)) 
    img, size, energie = biasDecomposing(img, LO_D, HI_D, cpt, deathZone, discretZone, 0, 0)
    
    return img,size,energie # return a decomposed image -> many sub-images (quadrants)
    # return plot, energy to plot
    
##### recompose Image without Wavelet #####

def recomposeImageWithoutWavelet(cpt, compressedImage):
    img = recomposer(compressedImage, LO_R, HI_R, cpt)
    return img # return the recomposed image
#
