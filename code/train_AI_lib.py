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
    print("Finished Conversion of Images")

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
        img = transform(img).float()
        return(imgName, img, self.imgDict[imgName], float(len(self.imgDict[imgName])))

class trainNet:
    def __init__(self, net, hyperParams, crit, optim, oPath="../saved/TrainingRuns", isCuda=1):
        self.net  = net
        self.oPath = oPath
        self.hyperParams = hyperParams
        self.isCuda = isCuda

        self.crit = crit()
        self.opti = optim(self.net.parameters(), self.hyperParams['lr'])

        self.trainLoad = utilData.DataLoader(self.net.trainData, batch_size=self.hyperParams['batch'], shuffle=True)
        self.valLoad   = utilData.DataLoader(self.net.valData  , batch_size=self.hyperParams['batch'], shuffle=True)
        self.testLoad  = utilData.DataLoader(self.net.testData , batch_size=self.hyperParams['batch'], shuffle=True)
    
    def train(self, draw=1):
        # Generating file name to save training run to
        hypKeyList = list(self.hyperParams.keys())
        hypValList = list(self.hyperParams.values())

        genName    = self.net.name+"_"
        for i in range(len(hypKeyList)-1): genName += hypKeyList[i]+str(hypValList[i])+"_"; i=len(hypKeyList)-1
        genName += hypKeyList[i]+str(hypValList[i])
        modelpath = self.oPath+"/"+genName
        
        try: os.makedirs(modelpath)  # Make the directory
        except FileExistsError: None
        except: print("Error Creating File"); return
        else: None
        
        torch.manual_seed(42)

        iters = [epoch for epoch in range(self.hyperParams["noEpoch"])]
        trainLosses, valLosses, trainAccs, valAccs = [], [], [], []
        for epoch in range(self.hyperParams["noEpoch"]):
            if self.isCuda and torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

            start.record()
            trainAcc, trainLoss = self.net.evaluate(self.net, self.trainLoad, criterion=self.crit, optimizer=self.opti, isTraining=1, gpu=self.isCuda)
            valAcc  , valLoss   = self.net.evaluate(self.net, self.valLoad  , criterion=self.crit, optimizer=self.opti, isTraining=0, gpu=self.isCuda)

            if self.isCuda and torch.cuda.is_available(): torch.cuda.synchronize()
            end.record()

            trainLosses += [trainLoss]; trainAccs += [trainAcc]  # Appending results
            valLosses   += [valLoss  ]; valAccs   += [valAcc  ]

            torch.save(self.net.state_dict(), modelpath+"model_epoch{}".format(epoch))
            print("Epoch {} | Time Taken: {:.2f}s | Train acc: {:.10f},\
                  Train loss: {:.10f} | Validation acc: {:.10f}, Validation loss: {:.10f}\
                   ".format(epoch, start.elapsed_time(end)*0.001, trainAccs[epoch], trainLosses[epoch], valAccs[epoch], valLosses[epoch]))
        
        if draw: self.drawResults(modelpath, iters, trainLosses, valLosses, trainAccs, valAccs)

    def drawResults(self, modelpath, iters, trainLosses, valLosses, trainAcc, valAcc):
        plt.plot(iters, trainAcc, '.-', label =  "Training")
        plt.plot(iters,   valAcc, '.-', label = "Validation")
        plt.title("Model Accuracy against Epoch No")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.legend(); plt.grid()
        plt.savefig(modelpath+"Accuracy Graph.png")
        plt.show()
        plt.cla()

        plt.plot(iters, trainLosses, '.-', label =   "Training")
        plt.plot(iters,   valLosses, '.-', label = "Validation")
        plt.title("Model Loss against Epoch No")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.legend(); plt.grid()
        plt.savefig(modelpath+"Loss Graph.png")
        plt.show()


def evalRegress(net, loader, criterion, optimizer, isTraining, gpu=1):
    lossTot = 0
    for _, img, _, noBbox in loader:  # if isTraining, computing loss and training, if not, then computing loss
        if gpu and torch.cuda.is_available: img = img.cuda(); noBbox = noBbox.cuda(); noBbox = noBbox.float()
        pred = net(img); pred=torch.squeeze(pred, 1)
        loss = criterion(pred, noBbox); lossTot += loss
        if isTraining:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    acc = []
    for _, img, _, noBbox in loader:  # Computing accuracy
        if gpu and torch.cuda.is_available: img = img.cuda(); noBbox = noBbox.cuda()
        pred = net(img)
        acc += [torch.sum((pred-noBbox)**2)]

    accuracy = sum(acc)/len(loader)
    avgLoss = lossTot/len(loader)
    return(accuracy, avgLoss)

# BASE CODE FOR a neural net module to push into trainNet class
class exNetClass(nn.Module):
    def __init__(self, name, evalF, datas):
        super(exNetClass, self).__init__()
        self.name = name
        self.evaluate = evalF
        self.trainData = datas[0]
        self.valData   = datas[1]
        self.testData  = datas[2]

        self.conv1 = nn.Conv2d(3,   15,  6, stride=2)  # n = 1024 -> 510
        self.conv2 = nn.Conv2d(15,  30,  6, stride=2)  # n = 510  -> 255
        self.pool1 = nn.MaxPool2d(3, 2)                # n = 255  -> 127
        self.conv3 = nn.Conv2d(30,  60,  6, stride=2)  # n = 127  -> 62
        self.pool2 = nn.MaxPool2d(4, 2)                # n = 62   -> 30

        self.fc1   = nn.Linear(29*29*60, 20)
        self.fc2   = nn.Linear(20, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(-1, 29*29*60)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return(x)

""" SAMPLE CODE TO RUN TRAINING MODULE
dictPath = "../saved/splitData/"
inPath   = "../data/working-wheat-data/train"

trainData = imgLoader(dictPath+"trainData", inPath)
valData   = imgLoader(dictPath+"valData"  , inPath)
testData  = imgLoader(dictPath+"testData" , inPath)

netA = exNetClass("netA", evalRegress, [trainData, valData, testData]); netA.cuda()
netATrain = trainNet(netA, {'noEpoch':10 ,'lr': 0.001, 'batch': 50}, nn.MSELoss, torch.optim.Adam)
netATrain.train()
"""

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