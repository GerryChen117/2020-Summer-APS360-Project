import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utilData
import torchvision
import torchvision.models
import matplotlib.pyplot as plt
import cv2
import os
import sys
import ast
import pandas as pd
from scipy.stats import norm

# ========== Main Training Code ========== #

def trainNet(net, data, batchsize, epochNo, lr, oPath="saved", mode='default', cuda=1, draw=1):
    """
    Big boy function that actually brings all of the function above together and actually trains the model
    Arguments:
        net      : the neural net object
        data     : a 2 list of data loaders; [trainLoader, valLoader]
        batchsize: the chosen batchsize
        epochNo  : the chosen max epochNo
        lr       : the chosen learning rate
        oPath    : root output path for all files. if '/root' is given, will save to '/root/TrainingRuns/<Folder>/
        mode     : string used to easily choose particular parameters such as criterion or optimizer
        cuda     : boolean to indicate if cuda should be used
        draw     : boolean to indicate if the graph should be drawn
    Returns:
        iters, trainLosses, valLosses, trainAcc, valAcc: for debug purposes. All lists of the values at each epoch
    """
    # Defining a saving path for ease of use
    if mode == 'default':
        # Define criterion and optimizers
        criterion    = nn.MSELoss()
        optimizer    = torch.optim.Adam(net.parameters(), lr=lr)
        evaluate     = evalRegress
        minibatch    = 0
        functionName = "RegAdamTrainer"  # Name of the function used (incase we decide to use different optimizers, use alexnet etc)

    elif mode == 'auto':
        criterion    = nn.CrossEntropyLoss()
        optimizer    = torch.optim.Adam(net.parameters(), lr=lr)
        evaluate     = evalAutoEnc
        minibatch    = 4
        functionName = "AutoEncTrainer"

    else: print("Unsupported Training Type"); return()

    modelpath = oPath+"/TrainingRuns/{}/{}_b{}_te{}_lr{}/".format(functionName, net.name, batchsize, epochNo, lr)
    torch.manual_seed(1000)
    try: os.makedirs(modelpath)  # Make the directory
    except FileExistsError: None
    except: print("Error Creating File"); return()
    else: None

    trainData, valData = data[0], data[1]  # Loading Required Data

    iters, trainLosses, valLosses, trainAcc, valAcc = [], [], [], [], []  # variables to graph and save
    for epoch in range(epochNo):
        if cuda and torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)

        if cuda and torch.cuda.is_available(): start.record()
        iters += [epoch]
        #evaluate(net=net, loader=trainData, criterion=criterion, optimizer=optimizer, isTraining=True)
        trainResults = evaluate(net=net, loader=trainData, criterion=criterion, optimizer=optimizer, isTraining=True , cuda=cuda, noBatches=0)  # Calculating training error and loss
        valResults   = evaluate(net=net, loader=  valData, criterion=criterion, optimizer=optimizer, isTraining=False, cuda=cuda, noBatches=minibatch)

        if cuda and torch.cuda.is_available(): end.record(); torch.cuda.synchronize()

        trainLosses += [trainResults[0]]; trainAcc += [trainResults[1]]  # Appending results
        valLosses   += [  valResults[0]]; valAcc   += [  valResults[1]]
        torch.save(net.state_dict(), modelpath+"model_epoch{}".format(epoch))

        if cuda and torch.cuda.is_available():
            print("Epoch {} | Time Taken: {:.2f}s | Training Error: {:.10f}, Training loss: {:.10f} | Validation Error: {:.10f}, Validation loss: {:.10f}".format(epoch, start.elapsed_time(end)*0.001, trainAcc[epoch], trainLosses[epoch], valAcc[epoch], valLosses[epoch]))
        else: 
            print("Epoch {} | Training Error: {:.10f}, Training loss: {:.10f} | Validation Error: {:.10f}, Validation loss: {:.10f}".format(epoch, trainAcc[epoch], trainLosses[epoch], valAcc[epoch], valLosses[epoch]))

    if draw: drawResults(modelpath, iters, trainLosses, valLosses, trainAcc, valAcc)
    return(iters, trainLosses, valLosses, trainAcc, valAcc)

# ========== Evaluation Functions ========== #
def evalRegress(net, loader, criterion, optimizer, isTraining, cuda=1, noBatches=0):
    """
    Function used in trainNet() to evaluate a given net for one epoch
    Arguments:
        net       : The net object
        loader    : the loader whose images are being put through the network for evaluation
        criterion : the criterion function
        optimizer : the optimizer function
        isTraining: Boolean to indicate if training should occur with evaluation, if True, optimizer will perform step
        cuda      : Boolean to indicate if cuda is to be utilized
        noBatches : the number of batches to iterate though; 0 is maximum of loader
    Returns:
        avgLoss : The calculated average loss over the entire epoch (root mean squared error)
        Accuracy: The calculated accuracy over the epoch (Percentage of correct predictions)
    """
    lossTot = 0; correct = 0; total = 0  # Define key variables
    for i, (img, noBbox, _, _) in enumerate(loader):
        if cuda and torch.cuda.is_available(): img = img.detach().cuda();  noBbox = noBbox.detach().cuda()
        img = img.float(); noBbox = noBbox.float()
        pred = net(img); pred=torch.squeeze(pred, 1)
        loss = criterion(pred, noBbox); lossTot += float(loss)
        if isTraining:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
        if noBatches!=0 and i==noBatches: break
        correct += torch.round(pred).eq(noBbox).sum().item()
        total   += noBbox.size()[0]

        pred = pred.detach()

    accuracy = 1-(correct/total)
    avgLoss = np.sqrt(lossTot/len(loader))
    return(avgLoss, accuracy)

def evalAutoEnc(net, loader, criterion, optimizer, isTraining, cuda=1, noBatches=0):
    """
    Evaluation function used in trainnet to specify the training loop of an auto encoder
    Arguments:
        net       : The net object
        loader    : the loader whose images are being put through the network for evaluation
        criterion : the criterion function
        optimizer : the optimizer function
        isTraining: Boolean to indicate if training should occur with evaluation, if True, optimizer will perform step
        cuda      : Boolean to indicate if cuda is to be utilized
        noBatches : the number of batches to iterate though; 0 is maximum of loader
    Returns:
        Accuracy  : The calculated accuracy over the epoch
        avgLoss   : The calculated average loss over the entire epoch
    """
    correct = 0
    lossTot = 0
    total   = 0
    softMax = nn.LogSoftmax()
    for i, (img, compImg, _) in enumerate(loader):  # if isTraining, computing loss and training, if not, then computing loss
        if cuda and torch.cuda.is_available(): img = img.detach().cuda(); compImg = compImg.detach().cuda()
        pred = net(img)
        loss = criterion(pred, compImg); lossTot += float(loss)
        if isTraining:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if noBatches!=0 and i==noBatches: break

        pred     = softMax(pred)
        pred     = pred.max(1, keepdim=True)[1]
        correct += pred.eq(compImg.view_as(pred)).sum().item()
        total   += compImg.size()[0]*compImg.size()[1]*compImg.size()[2]
        loss = loss.detach(); pred = pred.detach()

    accuracy = 1-(correct/total)
    avgLoss  = lossTot/len(loader)
    return(avgLoss, accuracy)

# ========= Image Loaders ========== #

class imgLoader(utilData.Dataset):
    """
    Custom pytorch dataset for loading images
    __init__:
        Arguments:
            dataPath: path to dictionary created by splitData()
            imgPath : path to an image folder; note, as the class functions by looking up the file name, different folder directories can be given
                      with the same dataPath as long as the images in those folders have a the same name as the original folder.
    __len__ :
        Function which is called when one uses the len() function on an imgLoader Object, returns the number of images
    __getitem__(self, idx):
        Function as required by a Map-style dataset. https://pytorch.org/docs/stable/data.html#map-style-datasets
        returns:
            img     : A tensor version of the image
            noBbox  : The number of bboxes for this given image
            imgName : name of image (for debug purposes)
            bboxList: list of bounding boxes as defined as [[bbox1], [bbox2], ...] (for debug purposes)
    """
    def __init__(self, dataPath, imgPath, func=None):
        # Defining Variables Required to find and load the data
        self.imgDict  = torch.load(dataPath)
        self.imgPath  = imgPath  + "/"
        self.func     = func

        # Defining required pytorch objects
        self.trans    = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __len__(self):
        return(len(self.imgDict))

    def __getitem__(self, idx):
        # Load requested image and convert to pytorch tensor
        imgName = list(self.imgDict.keys())[idx]
        img     = cv2.imread(self.imgPath+imgName)
        if self.func != None: img = self.func(img)

        img = self.trans(img).float()

        return(img, float(len(self.imgDict[imgName])), imgName, self.imgDict[imgName])

class tensorLoader(utilData.Dataset):
    """
    Custom pytorch dataset for loading images
    __init__:
        Arguments:
            dataPath: path to dictionary created by splitData()
            imgPath : path to the folder containing tensors; note, as the class functions by looking up the file name, different folder directories can be given
                      with the same dataPath as long as the images in those folders have a the same name as the original folder.
    __len__ :
        Function which is called when one uses the len() function on an imgLoader Object, returns the number of images
    __getitem__(self, idx):
        Function as required by a Map-style dataset. https://pytorch.org/docs/stable/data.html#map-style-datasets
        returns:
            img     : A tensor version of the image
            noBbox  : The number of bboxes for this given image
            imgName : name of image (for debug purposes)
            bboxList: list of bounding boxes as defined as [[bbox1], [bbox2], ...] (for debug purposes)
    """
    def __init__(self, dataPath, imgPath):
        # Defining Variables Required to find and load the data
        self.imgDict  = torch.load(dataPath)
        self.imgPath  = imgPath  + "/"

    def __len__(self):
        return(len(self.imgDict))

    def __getitem__(self, idx):
        imgName = list(self.imgDict.keys())[idx]  # Get image name
        img = torch.load(self.imgPath+imgName.split('.jpg')[0])  # load from given file
        return(img, float(len(self.imgDict[imgName])), imgName, self.imgDict[imgName])

class alexLoader(utilData.DataLoader):
    """
    Custom pytorch dataset for loading images
    __init__:
        Arguments:
            dataPath: path to dictionary created by splitData()
            imgPath : path to an image folder; note, as the class functions by looking up the file name, different folder directories can be given
                      with the same dataPath as long as the images in those folders have a the same name as the original folder.
            cuda    : boolean to specify whether cuda should be used for alexnet
    __len__ :
        Function which is called when one uses the len() function on an imgLoader Object, returns the number of images
    __getitem__(self, idx):
        Function as required by a Map-style dataset. https://pytorch.org/docs/stable/data.html#map-style-datasets
        returns:
            img     : A tensor version of the image
            noBbox  : The number of bboxes for this given image
            imgName : name of image (for debug purposes)
            bboxList: list of bounding boxes as defined as [[bbox1], [bbox2], ...] (for debug purposes)
    """
    def __init__(self, dataPath, imgPath, cuda=1):
        # Defining Variables Required to find and load the data
        self.imgDict  = torch.load(dataPath)
        self.imgPath  = imgPath  + "/"
        self.cuda     = cuda

        # Defining required pytorch objects
        self.trans    = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.alex     = torchvision.models.alexnet(pretrained=True)
        if torch.cuda.is_available() and self.cuda: self.alex.cuda()

    def __len__(self):
        return(len(self.imgDict))

    def __getitem__(self, idx):
        # Load requested image and convert to pytorch tensor
        imgName = list(self.imgDict.keys())[idx]
        img     = cv2.imread(self.imgPath+imgName)
        img     = self.trans(img).float()
        if torch.cuda.is_available() and self.cuda: img = img.cuda()

        # push image into alexnet features
        img = torch.squeeze(self.alex.features(torch.unsqueeze(img, 0)), 0)
        img = img.detach().cpu()

        return(img, float(len(self.imgDict[imgName])), imgName, self.imgDict[imgName])

class autoLoader(utilData.DataLoader):
    """
    Custom pytorch dataset for loading images
    __init__:
        Arguments:
            dataPath: path to dictionary created by splitData()
            imgPath : path to an image folder; note, as the class functions by looking up the file name, different folder directories can be given
                      with the same dataPath as long as the images in those folders have a the same name as the original folder.
    __len__ :
        Function which is called when one uses the len() function on an imgLoader Object, returns the number of images
    __getitem__(self, idx):
        Function as required by a Map-style dataset. https://pytorch.org/docs/stable/data.html#map-style-datasets
        returns:
            img: A tensor version of the image
            compImg: A tensor version of the bbox
            imgName: the name of the image for debug purposes
    """
    def __init__(self, dataPath, imgPath):
        # Defining Variables Required to find and load the data
        self.imgDict = torch.load(dataPath)
        self.imgPath = imgPath  + "/"

        # Defining required pytorch objects
        self.trans   = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __len__(self):
        return(len(self.imgDict))

    def __getitem__(self, idx):
        imgName = list(self.imgDict.keys())[idx]
        img     = cv2.imread(self.imgPath+imgName)
        compImg = createMask(self.imgDict[imgName]).copy()

        img     = self.trans(img).float()
        compImg = torch.squeeze(self.trans(compImg).long(), 0)

        return(img, compImg, imgName)

def loadData(batchsize, dictPath = "saved/splitData", inPath = "data/working-wheat-data/train", mode='default', args={'func':None,'cuda':1}):
    """
    Function to quickly batch generate a DataLoader
    Arguments:
        batchsize: requested batchsize
        dataPath : path to dictionary created by splitData()
        inPath   : path to an image folder; note, as the class functions by looking up the file name, different folder directories can be given
                      with the same dataPath as long as the images in those folders have a the same name as the original folder.
        alexnet  : bool to pass into imgLoader, to tell it that it if it is loading alexnet features
    Returns:
        trainLoader, valLoder, testLoader: The DataLoaders batched as reqested
    """
    if mode == 'default':
        trainData = imgLoader(dataPath=dictPath+"/trainData", imgPath=inPath, func=args['func'])
        valData   = imgLoader(dataPath=dictPath+"/valData"  , imgPath=inPath, func=args['func'])
        testData  = imgLoader(dataPath=dictPath+"/testData" , imgPath=inPath, func=args['func'])

    elif mode == 'tensor':
        trainData = tensorLoader(dataPath=dictPath+"/trainData", imgPath=inPath)
        valData   = tensorLoader(dataPath=dictPath+"/valData"  , imgPath=inPath)
        testData  = tensorLoader(dataPath=dictPath+"/testData" , imgPath=inPath)

    elif mode == 'alex':
        trainData = alexLoader(dataPath=dictPath+"/trainData", imgPath=inPath, cuda=args['cuda'])
        valData   = alexLoader(dataPath=dictPath+"/valData"  , imgPath=inPath, cuda=args['cuda'])
        testData  = alexLoader(dataPath=dictPath+"/testData" , imgPath=inPath, cuda=args['cuda'])       

    elif mode == 'auto':
        trainData = autoLoader(dataPath=dictPath+"/trainData", imgPath=inPath)
        valData   = autoLoader(dataPath=dictPath+"/valData"  , imgPath=inPath)
        testData  = autoLoader(dataPath=dictPath+"/testData" , imgPath=inPath)

    trainLoader = utilData.DataLoader(trainData, batch_size=batchsize, shuffle=1)
    valLoader   = utilData.DataLoader(valData  , batch_size=batchsize, shuffle=1)
    testLoader  = utilData.DataLoader(testData , batch_size=batchsize, shuffle=1)

    return(trainLoader, valLoader, testLoader)

# ========== Functions related to data processing, converting or transforming images ========= #

def splitData(ratio=[0.8, 0.1, 0.1], iPath="data/working-wheat-data/train", oPath="saved/splitData", koPath="data/working-wheat-data/train.csv"):
    """
    Function that takes the given input path (iPath) splits the image set into a given ratio then saves the names of images in each list to files
    output to output path (oPath)

    Arguments:
        ratio: a len-3 list containing the ratios of training, validation and testing data. default = [0.8, 0.1, 0.1]
        iPath: input path. default = "data/working-wheat-data/train"
        oPath: output path. default = "saved/splitData"
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

def openCVImgConvert(func, oPath, iPath="data/working-wheat-data/train"):
    """
    Funtion to help quickly apply an openCV image transformation and save the outputs
    Examples of Open CV features:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html#canny
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html#gradients
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#morphological-ops
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#filtering
    Arguments:
        func : a function which takes in an image, transforms it, and then returns the np array
        oPath: the folder to which the new images are saved to.
        iPath: the folder to the original images.
    """

    files = [f for f in os.listdir(iPath) if os.path.isfile(os.path.join(iPath, f))]  # Creating a list of files in iPathDirectory
    try: os.makedirs(oPath)  # Make the requested oPath
    except FileExistsError: None
    except: print("<openCVImgConvert> error creating folder {}".format(oPath)); return(0)
    else: None
    
    for i, f in enumerate(files):  # apply the given func() to every image in file
        cv2.imwrite(oPath+"/"+f, func(cv2.imread(iPath+"/"+f)))	
        print("Converted {:.2f}% of images".format(100*i/len(files)), end='\r')
    print("Finished Conversion of Images")

def createMask(bboxes, imgRes=(1024, 1024)):
    globalMask = np.zeros(imgRes, dtype=bool)
    for bbox in bboxes:
        oneMask = np.zeros(imgRes, dtype=bool)
        bbox = [int(number) for number in bbox]
        # Don't ask me why this flip works, numpy and cv2 take images in differently and it's a pain to try and figure out the transformations    
        oneMask[bbox[0]:(bbox[0]+bbox[2]), imgRes[1]-(bbox[1]+bbox[3]):imgRes[1]-bbox[1]] = np.ones((bbox[2], bbox[3]), dtype=bool)
        globalMask = np.bitwise_or(globalMask, oneMask)
    globalMask = np.rot90(globalMask, 1)
    return(globalMask)

def genMaskedImg(net, outPath='saved/autEncMasked', cuda=1):
    trainData, valData, testData = loadData(1)
    softMax = nn.LogSoftmax()

    try: os.makedirs(outPath+"/"+net.name)  # Make the requested oPath
    except FileExistsError: None
    except: print("<openCVImgConvert> error creating folder {}".format(oPath)); return(0)
    else: None

    for data in [trainData, valData, testData]:
        if data == trainData: print("Convering Training Images"  )
        if data == valData  : print("Convering Validation Images")
        if data == testData : print("Convering Testing Images"   )
        for i, (img, _, imgName, _) in enumerate(data):
            if torch.cuda.is_available() and cuda: img = img.cuda()
        
            pred = softMax(net(img)).cpu().detach()
            pred = pred.max(1, keepdim=True)[1].numpy()
            
            img  = torch.squeeze(img, 0)
            img  = torch.transpose(img, 0, 1)
            img  = torch.transpose(img, 1, 2)
            img  = img.cpu().numpy()

            img[:, :, 0] = np.multiply(img[:, :, 0], pred)
            img[:, :, 1] = np.multiply(img[:, :, 1], pred)
            img[:, :, 2] = np.multiply(img[:, :, 2], pred)

            plt.imsave(outPath+"/"+net.name+"/"+imgName[0], img)
            print("Converted {:.2f}%".format(100*i/len(data)), end='\r')


# ========== NON-ESSENTIAL HELPER FUNCTIONS ========== #
def drawResults(modelpath, iters, trainLosses, valLosses, trainAcc, valAcc):
    """
    Function used to quickly graph the results of training
    Arguments:
        modelpath             : path to save the image to
        iters                 : list of epoch numbers
        trainLosses, valLosses: lists of calculated losses per epoch
        trainAcc, valAcc      : lists of calculated accuracies per epoch
    """
    plt.plot(iters, trainAcc, '.-', label =  "Training")
    plt.plot(iters,   valAcc, '.-', label = "Validation")
    plt.title("Model Error against Epoch No")
    plt.xlabel("Epoch"); plt.ylabel("Error")
    plt.legend(); plt.grid()
    plt.savefig(modelpath+"Error Graph.png")
    plt.show()
    plt.cla()

    plt.plot(iters, trainLosses, '.-', label =   "Training")
    plt.plot(iters,   valLosses, '.-', label = "Validation")
    plt.title("Model Loss against Epoch No")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid()
    plt.savefig(modelpath+"Loss Graph.png")
    plt.show()

def prevImages(dataPath="saved/splitData/trainData", imgFolder="data/working-wheat-data/train"):
    """
    Function to simply test images and the bboxes
    Arguments:
        dataPath: path to csv file
        imgFolder: path to images
    """
    imgDict = torch.load(dataPath)  # Load Dictionary as generated by splitData
    for i, (name, bboxs) in enumerate(imgDict.items()):
        img = cv2.imread(imgFolder+"/"+name)  # read from image folder the image requested
        # Add bboxes
        [cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0,0,255),3) for bbox in bboxs]
        cv2.imshow('image', img)  # Show bboxes
        cv2.waitKey(0)  # wait for key press before moving to next image
        if i > 20: break  # Break after 20 images

def showResults(net, path):
    """
    Function that takes a given autoencoder, the path to a pre-trained state dict of that autoencoder and then displays a collage of images
    from left to right: displayes the orignal image, the bboxes given for that image, and the output from the network
    Arguments:
        net:  torch net obeject for an auto encoder
        path: path to the fully trained network
    """
    trainLoader, valLoader, testLoader = loadData(1, mode='auto')
    net.load_state_dict(torch.load(path))
    softMax = nn.LogSoftmax()
    for img, compImg, _ in trainLoader:
        out = softMax(net(img.cuda()))
        pred = out.max(1, keepdim=True)[1]
        pred = pred.cpu().numpy()

        img = torch.transpose(torch.squeeze(img), 0, 1)
        img = torch.transpose(torch.squeeze(img), 1, 2)

        c = np.zeros((1024, 1024, 3))
        c[:, :, 1] = compImg

        p = np.zeros((1024, 1024, 3))
        p[:, :, 2] = pred

        plt.imshow(np.concatenate((img, c, p), 1))
        break

def regresAnalysis(net, loader, modelpath, mode='train'):
    errorList = []
    noBboxList = []
    
    if mode   == 'train':
        oName = 'train_dist_graph.png'
        eType = 'Training'
    elif mode == 'val':
        oName = 'val_dist_graph.png'
        eType = 'Validation'
    elif mode == 'test':
        eType = 'Testing'
        oName = 'test_dist_graph.png'
    else:
        print("INVALID MODE")
        return


    for i, (img, noBbox, _, _) in enumerate(loader):
        pred  = net(img.cuda()).detach().cpu()
        pred  = torch.squeeze(pred, 0)
        error = torch.sqrt((pred-noBbox)**2).item()
        if pred < noBbox: error = -error
        errorList.append(error)
        noBboxList.append(noBbox.item())
        print("Tested {:.2f}%".format(100*i/len(loader)), end='\r')
        
    avg = np.average(noBboxList)

    unique      = np.unique(np.round(errorList))
    (mu, sigma) = norm.fit(errorList)

    color = 'tab:green'
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Error (units of wheat heads)')
    ax1.set_ylabel('Number of Images', color=color)
    n, bins, batches = ax1.hist(errorList, unique, facecolor=color)
    ax1.tick_params(axis='y', color=color)

    color = 'tab:red'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Percentage of Images', color=color)
    y = norm.pdf(bins, mu, sigma)
    ax2.plot(bins, y, color=color, linewidth=2)
    ax2.tick_params(axis='y', color=color)

    plt.title("Histogram and Normal Distribution of "+eType+" Errors\nμ={:.3f}, σ={:.3f}, avg no of heads={:.3f}".format(mu, sigma, avg))
    plt.savefig(modelpath+"/"+oName)

    return(errorList, mu, sigma, avg)


def calcNoParam(net):
    """
    Function to quickly find the number of total paramters of a net
    Parmeters:
        net: the neural net object
    """
    n = 0
    for param in net.parameters():
        a = 1
        for x in param.size():
            a *= x
        n+=a
    print(n)