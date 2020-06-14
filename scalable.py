from scipy import misc			#version 1.1.0
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from skimage import util

def spatialScalable(img, smallFrame, mediumFrame, largeFrame):
	image = Image.open(img)	#open the input image
	
	# scaling
	smallImage  = image.resize(smallFrame)		# generate small image
	mediumImage = image.resize(mediumFrame)		# generate medium image
	largeImage  = image.resize(largeFrame)		# generate large image
	
	smallImage.save("1.jpg")
	mediumImage.save("2.jpg")
	largeImage.save("3.jpg")
	
	# store in numpy array
	np.save("smallImage.npy", np.array(smallImage, dtype = np.float32))
	np.save("mediumImage.npy", np.array(mediumImage, dtype = np.float32))
	np.save("largeImage.npy", np.array(largeImage, dtype = np.float32))

	# return a dict
	return {"small":np.array(smallImage, dtype = np.float32), 
				"medium": np.array(mediumImage, dtype = np.float32), 
					"high": np.array(largeImage, dtype = np.float32)}

def simpleQualityScalable(img, lowNoise, mediumNoise, highNoise):
	image = misc.imread(img, mode='L')

	#add noise by piling many image
	low_image 	 = image * lowNoise
	medium_image = image * mediumNoise
	high_image 	 = image * highNoise

	return {"low": low_image, 
				"medium": medium_image, 
					"high": high_image}

def qualityScalable(img, lowNoise, mediumNoise, highNoise):
	image = Image.open(img)
	image = np.array(image)
	
	# add gaussian noise (3 levels)
	noise_low_gaussian    = util.random_noise(image,mode="gaussian", var=lowNoise)
	noise_medium_gaussian = util.random_noise(image,mode="gaussian", var=mediumNoise)
	noise_high_gaussian   = util.random_noise(image,mode="gaussian", var=highNoise)

	return {"low":noise_low_gaussian, 
				"medium": noise_medium_gaussian, 
					"high": noise_high_gaussian}

if __name__ == '__main__':
	image = "cameraman.tif"

	qCif = (176, 144) 
	Cif  = (352, 288)
	fCif = (704, 576)

	spatialDict = spatialScalable(image, qCif, Cif, fCif)
	simpleQualityDict = simpleQualityScalable(image, 2, 4, 8)
	qualityDict = qualityScalable(image, 2, 4, 8)


	# print(np.load("smallImage.npy"))
	# 3 lines, 4 columns
	f, axarr = plt.subplots(3, 4)

	# load spatial
	axarr[0,0].imshow(Image.open(image), cmap=plt.get_cmap('gray'))
	axarr[0,0].set_title("No resize")
	axarr[0,1].imshow(spatialDict.get("small"), cmap=plt.get_cmap('gray'))
	axarr[0,1].set_title("QCIF")
	axarr[0,2].imshow(spatialDict.get("medium"), cmap=plt.get_cmap('gray'))
	axarr[0,2].set_title("CIF")
	axarr[0,3].imshow(spatialDict.get("high"), cmap=plt.get_cmap('gray'))
	axarr[0,3].set_title("4CIF")

	# load simple quality
	axarr[1,0].imshow(Image.open(image), cmap=plt.get_cmap('gray'))
	axarr[1,0].set_title("No noise")
	axarr[1,1].imshow(simpleQualityDict.get("low"), cmap=plt.get_cmap('gray'))
	axarr[1,1].set_title("Low noise")
	axarr[1,2].imshow(simpleQualityDict.get("medium"), cmap=plt.get_cmap('gray'))
	axarr[1,2].set_title("Medium noise")
	axarr[1,3].imshow(simpleQualityDict.get("high"), cmap=plt.get_cmap('gray'))	
	axarr[1,3].set_title("High noise")

	# load gaussian noise
	axarr[2,0].imshow(Image.open(image), cmap=plt.get_cmap('gray'))
	axarr[2,1].imshow(qualityDict.get("low"), cmap=plt.get_cmap('gray'))
	axarr[2,2].imshow(qualityDict.get("medium"), cmap=plt.get_cmap('gray'))
	axarr[2,3].imshow(qualityDict.get("high"), cmap=plt.get_cmap('gray'))
	
	plt.show()