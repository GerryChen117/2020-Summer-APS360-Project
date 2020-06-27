import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utilData
import torchvision
import matplotlib.pyplot as plt
import cv2
import os
import ast
import pandas as pd

def splitData(ratio=[0.8, 0.1, 0.1], iPath="../data/working-wheat-data/train", oPath="../saved/splitData", koPath="../data/working-wheat-data/train.csv"):
    """
    Function that takes the given input path (iPath) splits the image set into a given ratio then saves the names of images in each list to files
    output to output path (oPath)

    Arguments:
        ratio: a len-3 list containing the ratios of training, validation and testing data. default = [0.8, 0.1, 0.1]
        iPath: input path. default = "../data/working-wheat-data/train"
        oPath: output path. default = "../saved/splitData"
        koPath: known bbox output csv path
    Returns:
        boolean of 0 or 1. fail or success.
    Creates Files:
        trainData
        valData
        testData

        files are saved via torch.save. each contains a dictionary.
        Keys of dictionaries are file names
        Values of dictionaries are a list of bboxes in the form [[bbox1], [bbox2], ...]
    """

    if sum(ratio) != 1:
        print("<splitData> error in ratio: does not sum to 1")
    if len(ratio) != 3:
        print("<splitData> error in ratio: input must be of length 3")

    try: os.makedirs(oPath)  # Making directory w/ error check
    except FileExistsError: None
    except: print("<splitData> error creating folder {}".format(oPath)); return(0)
    else: None

    np.random.seed(1234)  # Shuffling the data
    files = [f for f in os.listdir(iPath) if os.path.isfile(os.path.join(iPath, f))]
    np.random.shuffle(files)

    # Splitting images
    tLen = len(files); trainIndex=round(tLen*ratio[0])
    trainList=files[:trainIndex]; remain=files[trainIndex:]
    rLen = len(remain); valIndex=round(rLen*(ratio[1]/(ratio[1]+ratio[2])))
    valList=remain[:valIndex]; testList=remain[valIndex:]

    # append .csv info into dictionary
    trainDict = appendKnownOutputs(trainList, koPath)
    valDict   = appendKnownOutputs(valList  , koPath)
    testDict  = appendKnownOutputs(testList , koPath)

    # Save dictionary
    torch.save(trainDict, oPath+"/trainData")
    torch.save(valDict  , oPath+"/valData"  )
    torch.save(testDict , oPath+"/testData" )

    return(1)

def appendKnownOutputs(imgList, koPath):
    """
    helper function that takes a list of image file names, finds them in the train.csv file, and then creates a dictionary based on the results
    key of dictionary is the file name, value of dictionary is a list of bbox lists [[bbox1], [bbox2], ...]
    
    Arguments:
        imgList: List of image files names
        koPath : Path to the CSV file

    Returns:
        the dictionary as described above
    """
    imgDict = {}  # predefine dictionary
    db = pd.read_csv(koPath, header=0)  # Open csv
    for img in imgList:
        imgN = img.split(".jpg")[0]
        mask = db['image_id'].isin([imgN])  # Create a mask for the specific image name
        relRow = db.loc[mask]  # Find entries only with the specific image name
        imgDict[img] = [ast.literal_eval(bbox) for bbox in relRow['bbox']]  # Save all bboxes to dictionary
    return(imgDict)

def prevImages(dataPath="../saved/splitData/trainData", imgFolder="../data/working-wheat-data/train"):
    """
    Function to simply test images and the bboxes
    Arguments:
        dataPath: path to csv file
        imgFolder: path to images
    """
    imgDict = torch.load(dataPath) 
    for i, (name, bboxs) in enumerate(imgDict.items()):
        img = cv2.imread(imgFolder+"/"+name)
        [cv2.rectangle(img,(int(bbox[0]), int(bbox[1])),(int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])),(0,0,255),3) for bbox in bboxs]
        cv2.imshow('image', img)
        cv2.waitKey(0)
        if i > 20: break

class imgLoader(utilData.Dataset):
    """
    Custom pytorch dataset for loading images
    __init__:
        Arguments:
            dataPath: path to dictionary created by splitData()
            imgPath : path to an image folder; note, as the class functions by looking up the file name, different folder directories can be given
                      with the same dataPath as long as the images in those folders have a the same name as the original folder.
    """
    def __init__(self, dataPath, imgPath):
        self.imgDict = torch.load(dataPath)
        self.imgPath = imgPath + "/"
        self.keyList = list(self.imgDict.keys())
    def __len__(self):
        return(len(self.imgDict))
    def __getitem__(self, idx):
        imgName = self.keyList[idx]
        img = cv2.imread(self.imgPath+imgName)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        return(imgName, transform(img), self.imgDict[imgName])

def openCVImgConvert(func, oPath, iPath="../data/working-wheat-data/train"):
    """
    Funtion to help quickly apply an openCV image transformation and save the outputs
    Examples of Open CV features:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html#canny
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html#gradients
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#morphological-ops
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#filtering
    Arguments:
        func: a function which takes in an image, transforms it, and then returns the np array
        oPath: the folder to which the new images are saved to
        iPath: the folder to the original images
    """

    files = [f for f in os.listdir(iPath) if os.path.isfile(os.path.join(iPath, f))]
    try: os.makedirs(oPath)
    except FileExistsError: None
    except: print("<openCVImgConvert> error creating folder {}".format(oPath)); return(0)
    else: None
    
    for i, f in enumerate(files):
        cv2.imwrite(oPath+"/"+f, func(cv2.imread(iPath+"/"+f)))
        if i%200==0: print("Converted {:.2f}% of images".format(100*i/len(files)))

""" Eg (1)
### Use example for imageLoader

dictPath = "../saved/splitData/trainData"
inPath   = "../data/working-wheat-data/train"

loader   = imgLoader(dictPath, inPath)
for imgName, img, output in loader:
    ...
"""

""" Eg (2)
### Use example for openCVImgConvert

outPath = "../data/working-wheat-data/cv2_Canny_100_200"
inPath  = "../data/working-wheat-data/train"

edgeDetect = lambda oImg: cv2.Canny(oImg, 100, 200)  # The
openCVImgConvert(edgeDetect, outPath, inPath)  # Note: the images that are outputted have the same name as the original images, they are just in a different folder
"""

""" Eg (3)
### Use example for imageLoader, with the feature-detected images produced from Eg (2)

dictPath = "../saved/splitData/trainData"  # NOTE: DICT PATH IS THE SAME AS WAS IN Eg (1). IT DOES NOT NEED TO CHANGE
inPath   = "../data/working-wheat-data/cv2_Canny_100_200"  # ONLY THE IMAGE DIRECTORY HAS CHANGED

loader   = imgLoader(dictPath, inPath)
for imgName, img, output in loader:
    ...
"""

""" Eg (4)
### Use example of prevImages with converted images from Eg (2)
dictPath = "../saved/splitData/trainData"  # NOTE: DICT PATH IS THE SAME AS WAS IN Eg (1). IT DOES NOT NEED TO CHANGE
inPath   = "../data/working-wheat-data/cv2_Canny_100_200"  # ONLY THE IMAGE DIRECTORY HAS CHANGED

prevImages(dataPath=dictPath, imgFolder=inPath)
"""