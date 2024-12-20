import os
import time
import torch
import shutil
import numpy as np
import torch.nn as nn
import torch.utils.data as TorchData
from torch.utils.data import DataLoader, Dataset, Sampler
import ReadData as ReadData
import PrintAndPlot as PrintAndPlot
from CustomedDataset import TensorDatasetWithTransform,UltrasoundDataTransform2,UltrasoundDataTransform2_b,UltrasoundDataTransform_b,UltrasoundDataTransform
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.distributed import init_process_group
import torch.distributed as dist
class DummySampler(Sampler):
    def __init__(self, data, batch_size, n_gpus=2):
        self.num_samples = len(data)
        self.b_size = batch_size
        self.n_gpus = n_gpus
    def __iter__(self):
        ids = []
        for i in range(0, self.num_samples, self.b_size * self.n_gpus):
            ids.append(np.arange(self.num_samples)[i: i + self.b_size*self.n_gpus :self.n_gpus])
            ids.append(np.arange(self.num_samples)[i+1: (i+1) + self.b_size*self.n_gpus :self.n_gpus])
        return iter(np.concatenate(ids))
    def __len__(self):
        return self.num_samples

class MachineLearningModel:
    def __init__(self, earlyStoppingPatience, learnRate, batchSize):
        self.EarlyStoppingPatience = earlyStoppingPatience
        self.LearnRate = learnRate
        self.BatchSize = batchSize
        self.Epsilon = 1e-6
        self.BestStateDict = None
        self.SetupSeed(2223)

    def SetupSeed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def LoadStateDictionary(self, stateDictionary):
        self.Net.load_state_dict(stateDictionary, strict = True)

    def min_max_normalize(self, data):
        '''
        data: 待归一化的数据，类型为list或np.array
        '''
        min_val = np.min(data)
        max_val = np.max(data)
        norm_data = [(x - min_val) / (max_val - min_val) for x in data]
        return norm_data
    def LoadSet(self, set, fold, transform, mode_flag):
        setDataIndex = fold[set + "SetDataIndex"]
        aaa = fold[set + "SetLabel"]
        setLabel = torch.from_numpy(aaa).float()
        setDataset = TensorDatasetWithTransform([setLabel],
                                                fold["DataPath"],
                                                setDataIndex,
                                                transform = transform,
                                                mode_flag=mode_flag)
        
        setLoader = TorchData.DataLoader(setDataset, batch_size = self.BatchSize, num_workers = 0, shuffle=True)
        return setDataIndex, setLabel, setLoader

    def LoadData(self, fold1, fold2,mode_flag):
        if mode_flag =="B":
            self.TrainDataIndex, self.TrainLabel, self.TrainLoader = self.LoadSet("Train", fold1,UltrasoundDataTransform_b, mode_flag)
            self.ValidationDataIndex, self.ValidationLabel, self.ValidationLoader = self.LoadSet("Validation", fold2,UltrasoundDataTransform2_b, mode_flag)  #
        else:
            self.TrainDataIndex, self.TrainLabel, self.TrainLoader = self.LoadSet("Train", fold1,UltrasoundDataTransform, mode_flag)
            self.ValidationDataIndex, self.ValidationLabel, self.ValidationLoader = self.LoadSet("Validation", fold2,UltrasoundDataTransform2,mode_flag)  # UltrasoundDataTransform2



def EvaluateMachineLearningModel(modelClass,\
                                 saveFolderPath, trainPaths, testPaths,\
                                 earlyStoppingPatience = 30, learnRate = 0.00001, batchSize = 16,\
                                 name = None,ratio=0,mode_flag="B",numclass=2):
    folds = ReadData.ReadFolds(trainPaths)
    testFold = ReadData.ReadFolds(testPaths)
    saveFolderPath = os.path.join(saveFolderPath, modelClass.__name__ + ("" if name is None else name) + time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()))
    os.mkdir(saveFolderPath)
    validationAnswers = []
    testAnswers = []

    for i in range(3):  
        print("-------------------------------------------------------------")
        print("fold:", i)
        print("-------------------------------------------------------------")
        model = modelClass(earlyStoppingPatience=earlyStoppingPatience, learnRate=learnRate, batchSize=batchSize,mode_flag=mode_flag,yuce_level=yuce_level,numclass=numclass)
        model.Net = model.Net.cuda()
        model.LossFunction = model.LossFunction.cuda()
        fold1_train, fold2_val = folds.NextFold()
        model.LoadData(fold1_train, fold2_val,mode_flag) 
        validationAnswer, BestStateDict1 = model.Train(ratio,mode_flag)
        validationAnswers.append(validationAnswer)

        if mode_flag == "B":
            model.TestDataIndex, model.TestLabel, model.TestLoader = model.LoadSet("Test", testFold.GetWholeAsTest(),UltrasoundDataTransform2_b,mode_flag)
        elif mode_flag=="C":
            model.TestDataIndex, model.TestLabel, model.TestLoader = model.LoadSet("Test", testFold.GetWholeAsTest(),UltrasoundDataTransform2,mode_flag) 
        else:
             model.TestDataIndex, model.TestLabel, model.TestLoader = model.LoadSet("Test", testFold.GetWholeAsTest(),UltrasoundDataTransform2,mode_flag) 
        testAnswer, _ = model.Evaluate(model.TestLoader, BestStateDict1, ratio,mode_flag) 
        testAnswers.append(testAnswer)
        torch.save(model.BestStateDict, os.path.join(saveFolderPath, "Fold%dWeights.pkl" % i))
 
    PrintAndPlot.ClassificationPrintAndPlot(validationAnswers, testAnswers, saveFolderPath)
