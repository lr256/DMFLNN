import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from termcolor import colored
from PrintAndPlot import TaskClassificationAnswer
from DMFLNN import InceptionResNetV2,DMFLNN,BasicConvolution2D
from StatisticsUtils import ClassificationMetrics, BinaryClassificationMetric
from MachineLearningModel import MachineLearningModel, EvaluateMachineLearningModel
import time
import warnings
from sklearn.metrics import f1_score
from StatisticsUtils import CalculateAUC
warnings.filterwarnings("ignore")
class TaskModel(MachineLearningModel):
    def __init__(self, earlyStoppingPatience, learnRate, batchSize,mode_flag,yuce_level,numclass):
        super().__init__(earlyStoppingPatience, learnRate, batchSize)
        if mode_flag == "B":
            self.Net = InceptionResNetV2(numOfClasses=numclass)
            self.Net.Convolution1A = BasicConvolution2D(1, 32, kernelSize=3, stride=2)
        elif mode_flag == "C":
            self.Net = InceptionResNetV2(numOfClasses=numclass)
            self.Net.Convolution1A = BasicConvolution2D(3, 32, kernelSize=3, stride=2)
        else:
            self.Net = DMFLNN(numOfClasses=numclass)

        self.LossFunction = nn.CrossEntropyLoss()

    def Train(self,ratio,mode_flag):
        epoch = 0
        patience = self.EarlyStoppingPatience
        momentum = 0.9
        l2_decay = 5e-4
        optimizer = torch.optim.Adam(self.Net.parameters(), lr=self.LearnRate)
        numOfInstance = len(self.TrainLabel)
        minLoss = float(0x7FFFFFFF)
        maxAUC =0
        bestValidationAnswer = None

        trainLosses = []
        validationLosses = []

        while patience > 0:#for i in range(500):
            self.Net.train()
            epoch += 1
            runningLoss = 0.0
            for batchImage, batchLabel, _ in self.TrainLoader:#64,3,224,224
                batchImage = batchImage.float().cuda()
                batchLabel1 = batchLabel[:, 0].long().cuda()#良性or恶性
                batchLabel_linchuang = batchLabel[:, 1:].float().cuda()
                optimizer.zero_grad()
                if mode_flag == "B":
                    outputClass = self.Net.forward(batchImage, batchLabel_linchuang, ratio)#batchLabel_xingtai
                    loss = self.LossFunction(outputClass, batchLabel1)
                elif mode_flag == "C":  # 如果是CDFI模态
                    batchLabel_linchuang = batchLabel[:, 21:23].float().cuda()
                    outputClass = self.Net.forward(batchImage, batchLabel_linchuang, ratio)#batchLabel_xingtai
                    loss = self.LossFunction(outputClass, batchLabel1)
                else:  # 如果是混合模态
                    outputClass= self.Net.forward(batchImage, batchLabel_linchuang,ratio)
                    loss = self.LossFunction(outputClass, batchLabel1)

                loss = loss.mean()
                runningLoss += loss.item()
                loss.backward()
                optimizer.step()

            self.Net.eval()
            trainLoss = (runningLoss * self.BatchSize) / numOfInstance
            validationAnswer, validationLoss = self.Evaluate(self.ValidationLoader, None,xishu,ratio,mode_flag,yuce_level)

            trainLosses.append(trainLoss)
            validationLosses.append(validationLoss)
            print("Epoch %d:  (Patience left: %d )\ntrainLoss -> %.3f, valLoss -> %.3f" % (epoch, patience, trainLoss, validationLoss))
            print("Accuracy -> %f" % validationAnswer.Accuracy, end = ", ")
            print("Recall -> %f" % validationAnswer.Recall, end = ", ")
            print("Precision -> %f" % validationAnswer.Precision, end = ", ")
            print("Sensitivity -> %f" % validationAnswer.Sensitivity, end = ", ")
            print("Specificity -> %f" % validationAnswer.Specificity)
            print("f1_score -> %f" % validationAnswer.F1)
            print("PPV -> %f" % validationAnswer.PPV)
            print("NPV -> %f" % validationAnswer.NPV)
            validationAUC, validationFPR, validationTPR = CalculateAUC(validationAnswer.Outputs[:, 1].numpy(), validationAnswer.Labels.numpy(), needThreshold=True,multi=False)
            if maxAUC < validationAUC:
                patience = self.EarlyStoppingPatience
                minLoss = validationLoss
                maxAUC = validationAUC
                print("AUC ->{}; maxAUC->{}".format(validationAUC, maxAUC))
                bestValidationAnswer = validationAnswer
                self.BestStateDict = copy.deepcopy(self.Net.state_dict())
                print(colored("Better!!!!!!!!!!!!!!!!!!!!!!!!!!!", "green"))
            else:
                patience -= 1
                print("AUC ->{}; maxAUC->{}".format(validationAUC, maxAUC))
                print(colored("Worse!!!!!!!!!!!!!!!!!!!!!!!!!!!", "red"))

        bestValidationAnswer.TrainLosses = trainLosses
        bestValidationAnswer.ValidationLosses = validationLosses
        return bestValidationAnswer,self.BestStateDict

    def Evaluate(self, dataLoader, stateDictionary,ratio,mode_flag):
        self.Net.eval()
        answer = TaskClassificationAnswer()
        if stateDictionary is not None:
            self.LoadStateDictionary(stateDictionary)
        with torch.no_grad():
            numOfInstance = len(dataLoader.dataset)
            runningLoss = 0.0
            TP = [0]
            FP = [0]
            TN = [0]
            FN = [0]

            for batchImage, batchLabel, batchDataIndex in dataLoader:
                batchImage = batchImage.float().cuda()
                batchLabel1 = batchLabel[:, 0].long().cuda()  # 良性or恶性
                batchLabel_linchuang = batchLabel[:, 5:].float().cuda()  # long().cuda()临床特征标签

                if mode_flag == "B":
                    outputClass = self.Net.forward(batchImage, batchLabel_linchuang, ratio)#batchLabel_xingtai
                    loss = self.LossFunction(outputClass, batchLabel1)
                elif mode_flag == "C":  # 如果是单模态
                    batchLabel_linchuang = batchLabel[:, 21:23].float().cuda()
                    outputClass = self.Net.forward(batchImage, batchLabel_linchuang, ratio)#batchLabel_xingtai
                    loss = self.LossFunction(outputClass, batchLabel1)
                else:  # 如果是混合模态
                    outputClass,outputClass_cdfi,outputClass_us = self.Net.forward(batchImage, batchLabel_linchuang,ratio,flag_feature=0)
                    loss = self.LossFunction(outputClass, batchLabel1) + xishu * self.LossFunction(outputClass_cdfi,batchLabel1) +  (xishu) * self.LossFunction(outputClass_us, batchLabel1)
                loss = loss.mean()  # + loss_ori.mean()
                runningLoss += loss.item()
                BinaryClassificationMetric(outputClass, batchLabel1, TP, FP, TN, FN, 0)
                answer.Outputs = torch.cat((answer.Outputs, outputClass.softmax(dim=1).cpu()), dim=0)
                answer.Labels = torch.cat((answer.Labels, batchLabel1.float().cpu()), dim=0)
                answer.DataIndexes += batchDataIndex

            print("batchLabel1:{}".format(batchLabel1))

            answer.Accuracy, answer.Recall, answer.Precision, answer.Specificity, answer.Sensitivity, answer.PPV, answer.NPV, answer.F1 = ClassificationMetrics(
                    TP[0], FP[0], TN[0], FN[0], self.Epsilon)
            loss = (runningLoss * self.BatchSize) / numOfInstance
            if stateDictionary is not None:
                print("batchLabel1:{}".format(batchLabel1))
                print("Accuracy -> %f" % answer.Accuracy, end=", ")
                print("Recall -> %f" % answer.Recall, end=", ")
                print("Precision -> %f" % answer.Precision, end=", ")
                print("Sensitivity -> %f" % answer.Sensitivity, end=", ")
                print("Specificity -> %f" % answer.Specificity)
                print("PPV -> %f" % answer.PPV)
                print("NPV -> %f" % answer.NPV)
                print("f1_score -> %f" % answer.F1)

            return answer, loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--TrainFolderPath", help = "define the train data folder path", type = str)
    parser.add_argument("--TestFolderPath", help = "define the test data folder path", type = str)
    parser.add_argument("--TrainInfoPath", help = "define the train info path", type = str)
    parser.add_argument("--TestInfoPath", help = "define the test info path", type = str)
    parser.add_argument("--SaveFolderPath", help = "define the save folder path", type = str)
    parser.add_argument("--Name", help = "define the name", type = str)
    args = parser.parse_args()
    args.TrainFolderPath = "/media/lr/images"  # train
    args.TrainInfoPath = "./train.xlsx"#
    args.TestFolderPath = "/media/lr/images"   # test
    args.TestInfoPath = "./test.xlsx"#
    args.SaveFolderPath = "./save_model"
    args.Name = "Lymph_Inv2_B_C" + time.strftime("%Y-%m-%d %H.%M.%S",time.localtime())  # _getroi_lastliner
    EvaluateMachineLearningModel(TaskModel, \
                                 args.SaveFolderPath, (args.TrainFolderPath, args.TrainInfoPath),
                                 (args.TestFolderPath, args.TestInfoPath), earlyStoppingPatience = 20,batchSize=32,\
                                 name=args.Name, ratio=1,mode_flag="B+C",numclass=2)
