'''version 1 : Codage sans perte '''
''' Member : TRAN Thanh Duy - Master 1 ICONE '''

#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import struct
import os
import decompose as decompose
import numpy.linalg as nl
import JPEG.Functions as jpeg
from PIL import Image
from statistics import mean
import scalable as scala            # quality scalability and spatial scalability

### Store encoded image in binary file (I also store encoded image into numpy array > np.array)
### encode_image is a binary string (I will show you) >>> binary file
def saveEncodedImage(file, string):
    with open(file,'wb') as f:
        for k in range(0,len(string),8):
            f.write(struct.pack('c',bytes([int(string[k:k+8],2)])))
        f.write(struct.pack('c',bytes([len(string)-(int(len(string)/8)*8)]))) # convert infos to byte >> write to bin file
        
### Load encoded image from binary file to rebuild (I also load encoded image from numpy array >> np.load)
### decoded_image is a binary string
def loadEncodedImage(file):        
	with open(file,'rb') as f:
		Outstring = [bitMapping(int(x), 255) for x in f.read(os.path.getsize(file))] #convert a string of bit -> raw infos
		string2 = ""
		for x in range(len(Outstring)-2):
			string2 = string2 + Outstring[x]
		remainString = Outstring[len(Outstring)-2][(8-int(Outstring[(len(Outstring)-1):][0],2)):]
		Outstring = string2 + remainString
		return Outstring  # return raw infos in form of string

### convert a number of bits -> a binary string, e.g bitMapping(10, 20) -> 01010
def bitMapping(bitNumber, bitsMax):
    if(bitsMax < bitNumber):
        print("bitsMax to encode < bitNumbers >> Error to encoded")
    val = "{}".format(bitNumber)
    valMax = "{}".format(bitsMax)
    maxNum = "{}".format(bin(int(valMax,10))[2:])
    maxFormat = "{}".format(bin(int(val,10))[2:])
    bitNumber = ""
    for i in range(len(maxFormat),len(maxNum)):
        bitNumber = bitNumber + "0"
    bitNumber += maxFormat

    return bitNumber

### convert a string bit (encoded) -> a integer
def reverseBitMapping(bitNumber):
    return int(str(bitNumber),2)

### fourQuadrantSplit original image -> 4 quadrants, each quadrant is transformed into a numpy array
### return a dict
def fourQuadrantSplit(image):
    image1 = np.array([image[x][:int(len(image)/2)] for x in range(int(len(image)/2))]).astype(int)
    image3 = np.array([image[x][int(len(image)/2):] for x in range(int(len(image)/2))]).astype(int)
    image2 = np.array([image[x][int(len(image)/2):] for x in range(int(len(image)/2), len(image))]).astype(int)
    image4 = np.array([image[x][:int(len(image)/2)] for x in range(int(len(image)/2), len(image))]).astype(int)
    return {"first": image1, "second": image2,"third": image3, "fourth":image4}

################## --- Huffman tree activites ########################

### insert an element into huffman tree
#### input : a list + a element => return a list after inserting
def huffmanTreeInsert(list, element):
    rang = 0
    while(rang < len(list) and (element[0] > list[rang][0]) ):
        rang += 1
    list.insert(rang, element)
    return list

### build a huffman tree ==> print the tree
##### return a tree
def buildHTree(occursIndexList, occursValList):
    huffmanTree = [[y,x] for x,y in zip(occursIndexList,occursValList)]
    
    # build tree: in a Huffman tree, left side < right side (always)
    while(len(huffmanTree) != 1):
        if(huffmanTree[0][0] < huffmanTree[1][0]):  # left side < right side
            leftSide = huffmanTree[0]
            rightSide = huffmanTree[1]
        else:                                       # swap left and right
            leftSide = huffmanTree[1]
            rightSide = huffmanTree[0]
        huffmanTree.pop(0)         # pop first element
        huffmanTree.pop(0)         # next, pop the second element
        huffmanTree = huffmanTreeInsert(huffmanTree, [leftSide[0]+rightSide[0], leftSide, rightSide])
    print(huffmanTree)              # print huffman tree, e.g: [342 - root, [170 - root, [82, 367] - left side, [88, 360] - right side]
    return huffmanTree[0]

#### this method will return a boolean value after finding a pix in a string into a huffman tree
def huffmanTreeFinding(huffmanTree, pix, string):
    if(len(huffmanTree) > 0):
        if(len(huffmanTree) > 1 and not(isinstance(huffmanTree[1], list))):     # a leaf
            if(huffmanTree[1] == pix):
                return string
        else:   # 1 child
            _result1 = huffmanTreeFinding(huffmanTree[1], pix, string + "0")
            if(_result1 != "nil"):
                return _result1         # empty
            if(len(huffmanTree) > 2):   # 2 childs
                _result1 = huffmanTreeFinding(huffmanTree[2], pix, string + "1")
                if(_result1 != "nil"):
                    return _result1
    else:   # len(huffmanTree <= 0)
        print("Tree invalid")
    return "nil"
            
###### Huffman encoding ############
###### return an encoded image #######
def HuffmanEncode(image, inputSize):
    beginBitNumber = len(bitMapping(inputSize,inputSize))
    reshapedImg = np.reshape(image, (-1,len(image)*len(image)))[0] # reshape input image
    FramedImg = pd.DataFrame(reshapedImg)   # transformr 2D size-mutable (tabular data structure)
    
    occursList = FramedImg[0].value_counts() # len of tabular (horizontal)
    
    occursValList = occursList.values.tolist() 
    occursIndexList = occursList.index.tolist()
    
    huffmanTree = buildHTree(occursIndexList, occursValList)    #build a huffman tree using index list and value list

    indexMax = max(occursIndexList)
    maxVal = max(occursValList)
    indexBitNumber = len(bitMapping(indexMax, indexMax))
    valueBitNumber = len(bitMapping(maxVal, maxVal))
    
    finalString = bitMapping(indexBitNumber, inputSize)
    finalString += bitMapping(valueBitNumber, inputSize)
    finalString += bitMapping(beginBitNumber*3+(indexBitNumber+valueBitNumber)*len(occursIndexList), inputSize)

    for rang in range(len(occursIndexList)):
        finalString += bitMapping(occursIndexList[rang], indexMax)
        finalString += bitMapping(occursValList[rang], maxVal)

    res = [huffmanTreeFinding(huffmanTree, reshapedImg[x], "") for x in range(len(reshapedImg)) ]
    _result = "".join(map(str, res))
    _result = finalString + _result
    
    # print("Encoded string of Huffman:", _result)
    return _result  # result is a string -> store to bin file
    
###### Huffman decoding ########
########## return "original" image ###########
def HuffmanDecode(_result, inputSize):
    beginBitNumber = len(bitMapping(inputSize,inputSize))
    indexBit = reverseBitMapping(_result[:beginBitNumber])
    indexVal = reverseBitMapping(_result[beginBitNumber:beginBitNumber*2])
    minVal = reverseBitMapping(_result[beginBitNumber*2:beginBitNumber*3])
    
    occursIndexList = []
    occursValList = []
    for i in range(beginBitNumber*3, minVal,indexBit+indexVal):
        occursIndexList.append(reverseBitMapping(_result[i:i+indexBit]))
        occursValList.append(reverseBitMapping(_result[(i+indexBit):(i+indexBit+indexVal)]))
    
    huffmanTree = buildHTree(occursIndexList, occursValList)    #build a huffman tree using index list and value list
    
    image = []  # output is a list that I will use np.load() to load
    
    clonedHuffmanTree = huffmanTree     # clone a new huffman tree 
    bit = minVal
    while (bit < len(_result)):
        if(len(clonedHuffmanTree) > 0):
            if(len(clonedHuffmanTree) > 1):
                if(not(isinstance(clonedHuffmanTree[1], list)) ) :  # if fist element of huffman tree is not a list
                    image.append(clonedHuffmanTree[1])              # make a new list by appending image (list) this first element -> return image (list)
                    clonedHuffmanTree = huffmanTree                 # re-assign cloned tree by current tree -> if skip this line, the memory leaking by recursive
                else:
                    if(_result[bit] == "0"):
                        clonedHuffmanTree = clonedHuffmanTree[1]
                        bit += 1
                    else:
                        if(len(clonedHuffmanTree) > 2 and _result[bit] == "1"):
                            clonedHuffmanTree = clonedHuffmanTree[2]
                            bit += 1
    if(not(isinstance(clonedHuffmanTree[1], list)) ) :
        image.append(clonedHuffmanTree[1])
    
    return image # image is a list of integer value -> loaded with np.array


############ RLE algorithm is builded from Huffman algorithm ===> that means that I will modify Huffman algorithm above to implement RLE algorithm ###########
############ Horizontal Encode ###########
############## like Huffman Encode, this method will return an encoded image in form of a binary string ############
def HorizontalEncode(image, inputSize):
    image = np.reshape(image, (-1,len(image)*len(image)))[0]
    nbFact = 0
    pixList = []
    orderList = []
    i = 0
    while (i < len(image)):
        count = i
        while (i < len(image) - 1 and image[i] == image[i + 1]):
            i += 1
        pixList.append(int(image[i]))
        orderList.append(i-count+1)
        nbFact += i-count+1
        i += 1
    minVal = -min(pixList)
    pixList = [i+minVal for i in pixList]

    i = 0
    maxVal = 10*int(mean(orderList))
    while(i < len(pixList)):
        if(orderList[i] > maxVal):
            orderList.insert(i+1, orderList[i]-maxVal)
            orderList[i] = maxVal
            pixList.insert(i+1, pixList[i])
        i += 1
    beginBitNumber = len(bitMapping(inputSize,inputSize))
    indexMax = max(orderList)
    maxVal = max(pixList)
    indexBitNumber = len(bitMapping(indexMax, indexMax))
    valueBitNumber = len(bitMapping(maxVal, maxVal))
    stringFinale = bitMapping(indexBitNumber, inputSize)
    stringFinale += bitMapping(valueBitNumber, inputSize)
    stringFinale += bitMapping(minVal, inputSize)

    for j in range(len(orderList)):
        stringFinale += bitMapping(orderList[j], indexMax)
        stringFinale += bitMapping(pixList[j], maxVal)
    return stringFinale     # output is a binary string -> use np.load() to load

############ Horizontal Decode ###########
############## like Huffman Decode, this method will return a decoded image in form of list ############
def HorizontalDecode(_result, inputSize):
    beginBitNumber = len(bitMapping(inputSize,inputSize))
    indexBit = reverseBitMapping(_result[:beginBitNumber])
    indexVal = reverseBitMapping(_result[beginBitNumber:beginBitNumber*2])
    minVal = reverseBitMapping(_result[beginBitNumber*2:beginBitNumber*3])
    
    orderList = []
    pixList = []
    for i in range(beginBitNumber*3, len(_result),indexBit+indexVal):
        orderList.append(int(reverseBitMapping(_result[i:i+indexBit])))
        pixList.append(int(reverseBitMapping(_result[(i+indexBit):(i+indexBit+indexVal)]))-minVal)
    image = []
    for i in range(len(orderList)):
        for count in range(orderList[i]):
            image.append(pixList[i])
    
    return image    # output is a list of integer

######################################### END of implementation phase #####################################

### input size to encode => 32 bits encoding
inputSize = 1600000


def compress(image, multipleQuadrantRecursive):
	fourQuadrantSplitdImage = fourQuadrantSplit(image)                             # firsly, split an image > 4 quadrants
    
	if(multipleQuadrantRecursive == 1):
		string0 = HuffmanEncode(fourQuadrantSplitdImage.get("first"), inputSize)
		print("Huffman encode simple")
	else:
		nombreR = multipleQuadrantRecursive - 1
		string0, _ = compress(fourQuadrantSplitdImage.get("first"), nombreR)
		print("Huffman encode diff")

	string1 = HorizontalEncode(fourQuadrantSplitdImage.get("second"), inputSize)
	string2 = HorizontalEncode(fourQuadrantSplitdImage.get("third"), inputSize)
	string3 = HorizontalEncode(fourQuadrantSplitdImage.get("fourth"), inputSize)
	print("RLE encode")

	beginBitNumber = len(bitMapping(inputSize,inputSize))
    
	string = bitMapping(multipleQuadrantRecursive, 15)
	string += bitMapping(len(string0)+3*beginBitNumber, inputSize)
	string += bitMapping(len(string1)+len(string0)+3*beginBitNumber, inputSize)
	string += bitMapping(len(string2)+len(string1)+len(string0)+3*beginBitNumber, inputSize)
	string = string + string0 + string1 + string2 + string3
  
	return string, beginBitNumber

def decompress(string, beginBitNumber):
	multipleQuadrantRecursive = reverseBitMapping(string[:4])
	string = string[4:]
	nombreBitFinImage0 = reverseBitMapping(string[:beginBitNumber])
	nombreBitFinImage1 = reverseBitMapping(string[beginBitNumber:2*beginBitNumber])
	nombreBitFinImage2 = reverseBitMapping(string[beginBitNumber*2:3*beginBitNumber])
	nombreBitFinImage3 = len(string)
    
	if(multipleQuadrantRecursive == 1):	
		image0 = HuffmanDecode(string[3*beginBitNumber:nombreBitFinImage0], inputSize)
		image0 = np.reshape(image0, (int(math.sqrt(len(image0))),-1))
		print("Huffman decode simple")
	else:
		image0 = decompress(string[3*beginBitNumber:nombreBitFinImage0], beginBitNumber)
		print("Huffman decode diff")
	image1 = HorizontalDecode(string[nombreBitFinImage0:nombreBitFinImage1], inputSize)
	image2 = HorizontalDecode(string[nombreBitFinImage1:nombreBitFinImage2], inputSize)
	image3 = HorizontalDecode(string[nombreBitFinImage2:nombreBitFinImage3], inputSize)
	print("RLE decode") 
    
	image = image1
	image1 = np.reshape(image1, (int(math.sqrt(len(image1))),-1))
	image2 = np.reshape(image2, (int(math.sqrt(len(image2))),-1))
	image3 = np.reshape(image3, (int(math.sqrt(len(image3))),-1))
    
	image = np.concatenate([image0, image2], axis = 1)
	image1 = np.concatenate([image3, image1], axis = 1)
	image = np.concatenate([image, image1], axis = 0)
	return image

# number of quandrants = 4 ^ multipleQuadrantRecursive (e.g: 1 -> 4 quadrants, 2 -> 16 quadrants, 3 = 64 quadrants)
multipleQuadrantRecursive = 4
image = "Lena.jpg"

# calculate entropy and compression ratio
from scipy.stats import entropy
import JPEG.Functions as jpeg
import scipy.misc


if __name__ == '__main__':

    # image = "Lena.jpg"

    # Scalability
    qCif = (176, 144) 
    Cif  = (352, 288)
    fCif = (704, 576)

    spatialDict = scala.spatialScalable(image, qCif, Cif, fCif)
    simpleQualityDict = scala.simpleQualityScalable(image, 2, 4, 8)
    qualityDict = scala.qualityScalable(image, 2, 4, 8)

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

    # -------------------------------------------------------------------------------------------
    # deploy huffman and rle
    # binary file
    binaryFile = "Lena.bin"
    
    # # number of quandrants = 4 ^ multipleQuadrantRecursive (e.g: 1 -> 4 quadrants, 2 -> 16 quadrants, 3 = 64 quadrants)
    # multipleQuadrantRecursive = 4

    img, plot, energie = decompose.decomposeImageWithWavelet(multipleQuadrantRecursive, image, 100, 20)

    plt.imshow(img, cmap='gray')
    plt.show()
    string, beginBitNumber = compress(img, multipleQuadrantRecursive)
    saveEncodedImage(binaryFile, string)

    ### decode phase
    string = loadEncodedImage(binaryFile)
    rebuiltImg = decompress(string, beginBitNumber)
    rebuiltImg = decompose.recomposeImageWithoutWavelet(multipleQuadrantRecursive, rebuiltImg)
    
    # 2 ways
    # save rebuiltImaage -> jpg (using scipy.misc)
    scipy.misc.imsave('outfile.jpg', np.array(rebuiltImg, dtype = np.float32))

    # save rebuiltImage -> numpy array
    np.save("rebuiltImage.npy", np.array(rebuiltImg, dtype = np.float32))

    plt.imshow(rebuiltImg, cmap='gray')
    plt.show()

    # --------------- Evaluation ----------------> Entropy (Before and After) --------------->
    # entropy before encoding
    imgReshaped = Image.open(image)
    X = imgReshaped.size[0]
    Y = imgReshaped.size[0]
    imag = np.array(imgReshaped.getdata()).reshape(X,Y)

    sumDCT = np.array([np.zeros(8) for x in range(8)])
    for x in range(0,X,8):
        for y in range(0,Y,8):
            sumDCT+=abs(jpeg.DCT(imag[x:x+8,y:y+8]))
    sumDCT = sumDCT / np.sum(sumDCT)
    print("Entropy before encoding", entropy(np.reshape(sumDCT, (64))))

    # entropy after encoding
    
    # load by Image.open()
    #imgAfter = Image.open("outfile.jpg")
    
    # load from numpy array
    imgAfter = Image.fromarray(np.load("rebuiltImage.npy"))
    XAfter = imgAfter.size[0]
    YAfter = imgAfter.size[1]
    imagAfter = np.array(imgAfter.getdata()).reshape(XAfter,YAfter)

    sumDCTAfter = np.array([np.zeros(8) for x in range(8)])
    for x in range(0,XAfter,8):
        for y in range(0,YAfter,8):
            sumDCTAfter+=abs(jpeg.DCT(imagAfter[x:x+8,y:y+8]))
    sumDCTAfter = sumDCTAfter / np.sum(sumDCTAfter)
    print("Entropy after encoding", entropy(np.reshape(sumDCTAfter, (64))))

    print("Entropy BEFORE - Entropy AFTER", entropy(np.reshape(sumDCT, (64)))-entropy(np.reshape(sumDCTAfter, (64))))

    # --------------- Evaluation ----------------> Compression rate (Before and After) --------------->
    # image size before compressiong
    originalSize = os.path.getsize("Lena.jpg")
    
    # image size before decompressing
    afterSize = os.path.getsize("outfile.jpg")

    # compression rate (%)
    print("compression ratio : " + str(abs(100 - afterSize/originalSize * 100)) + "%")
    
    # --------------- Evaluation ----------------> Compression courbe using subplot (Before and After) --------------->
    img, plot2,energie2 = decompose.decomposeImageWithWavelet(multipleQuadrantRecursive, image, 100, 50)
    img, plot3,energie3 = decompose.decomposeImageWithWavelet(multipleQuadrantRecursive, image, 100, 100)
    img, plot4,energie4 = decompose.decomposeImageWithWavelet(multipleQuadrantRecursive, image, 100, 200)
    img, plot5,energie5 = decompose.decomposeImageWithWavelet(multipleQuadrantRecursive, image, 100, 500)
    img, plot6,energie6 = decompose.decomposeImageWithWavelet(multipleQuadrantRecursive, image, 100, 1000)
    
    plt.plot([plot,plot2,plot3,plot4,plot5,plot6], [energie,energie2,energie3,energie4,energie5,energie6], color="red", marker="o")
    plt.show()

    # finding histogram

    f1,a1 = plt.subplots(1,2)
    a1 = a1.ravel()

    # find histogram before
    listBefore,_ = np.histogram(Image.open(image), bins=256,range=(0,255))

    # find histogram before
    listAfter,_ = np.histogram(imgAfter, bins=256,range=(0,255))
 
    list = [listBefore, listAfter]
    for idx,ax in enumerate(a1):
    	ax.hist(list[idx], color="green")
    plt.show()

