import torch
import torch.nn as nn
class BasicConvolution2D(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize, stride, padding = 0):
        super().__init__()
        self.Convolution = nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding, bias = False)
        self.BatchNormalization = nn.BatchNorm2d(outChannels, eps = 0.001, momentum = 0.1, affine = True)
        self.ReLU = nn.ReLU(inplace = False)

    def forward(self, x):
        x = self.Convolution(x)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        return x

class Mixed5B(nn.Module):
    def __init__(self):
        super().__init__()
        self.Branch0 = BasicConvolution2D(192, 96, kernelSize = 1, stride = 1)
        self.Branch1 = nn.Sequential(
            BasicConvolution2D(192, 48, kernelSize = 1, stride = 1),
            BasicConvolution2D(48, 64, kernelSize = 5, stride = 1, padding = 2))
        self.Branch2 = nn.Sequential(
            BasicConvolution2D(192, 64, kernelSize = 1, stride = 1),
            BasicConvolution2D(64, 96, kernelSize = 3, stride = 1, padding = 1),
            BasicConvolution2D(96, 96, kernelSize = 3, stride = 1, padding = 1))
        self.Branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride = 1, padding = 1, count_include_pad = False),
            BasicConvolution2D(192, 64, kernelSize = 1, stride =1))

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        x2 = self.Branch2(x)
        x3 = self.Branch3(x)
        x = torch.cat((x0, x1, x2, x3), 1)
        return x

class Block35(nn.Module):
    def __init__(self, scale = 1.0):
        super().__init__()
        self.Scale = scale
        self.Branch0 = BasicConvolution2D(320, 32, kernelSize = 1, stride = 1)
        self.Branch1 = nn.Sequential(
            BasicConvolution2D(320, 32, kernelSize = 1, stride = 1),
            BasicConvolution2D(32, 32, kernelSize = 3, stride = 1, padding = 1))
        self.Branch2 = nn.Sequential(
            BasicConvolution2D(320, 32, kernelSize = 1, stride = 1),
            BasicConvolution2D(32, 48, kernelSize = 3, stride = 1, padding = 1),
            BasicConvolution2D(48, 64, kernelSize = 3, stride = 1, padding = 1))
        self.Convolution = nn.Conv2d(128, 320, kernel_size = 1, stride = 1)
        self.ReLU = nn.ReLU(inplace = False)

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        x2 = self.Branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.Convolution(out)
        out = out * self.Scale + x
        out = self.ReLU(out)
        return out

class Mixed6A(nn.Module):
    def __init__(self):
        super().__init__()
        self.Branch0 = BasicConvolution2D(320, 384, kernelSize = 3, stride = 2)
        self.Branch1 = nn.Sequential(
            BasicConvolution2D(320, 256, kernelSize = 1, stride = 1),
            BasicConvolution2D(256, 256, kernelSize = 3, stride = 1, padding = 1),
            BasicConvolution2D(256, 384, kernelSize = 3, stride = 2))
        self.Branch2 = nn.MaxPool2d(3, stride = 2)

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        x2 = self.Branch2(x)
        x = torch.cat((x0, x1, x2), 1)
        return x

class Block17(nn.Module):
    def __init__(self, scale = 1.0):
        super().__init__()
        self.Scale = scale
        self.Branch0 = BasicConvolution2D(1088, 192, kernelSize = 1, stride = 1)
        self.Branch1 = nn.Sequential(
            BasicConvolution2D(1088, 128, kernelSize = 1, stride = 1),
            BasicConvolution2D(128, 160, kernelSize = (1, 7), stride = 1, padding = (0, 3)),
            BasicConvolution2D(160, 192, kernelSize = (7, 1), stride = 1, padding = (3, 0)))
        self.Convolution = nn.Conv2d(384, 1088, kernel_size = 1, stride = 1)
        self.ReLU = nn.ReLU(inplace = False)

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.Convolution(out)
        out = out * self.Scale + x
        out = self.ReLU(out)
        return out

class Mixed7A(nn.Module):
    def __init__(self):
        super().__init__()
        self.Branch0 = nn.Sequential(
            BasicConvolution2D(1088, 256, kernelSize = 1, stride = 1),
            BasicConvolution2D(256, 384, kernelSize = 3, stride = 2))
        self.Branch1 = nn.Sequential(
            BasicConvolution2D(1088, 256, kernelSize = 1, stride = 1),
            BasicConvolution2D(256, 288, kernelSize = 3, stride = 2))
        self.Branch2 = nn.Sequential(
            BasicConvolution2D(1088, 256, kernelSize = 1, stride = 1),
            BasicConvolution2D(256, 288, kernelSize = 3, stride = 1, padding = 1),
            BasicConvolution2D(288, 320, kernelSize = 3, stride = 2))
        self.Branch3 = nn.MaxPool2d(3, stride = 2)

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        x2 = self.Branch2(x)
        x3 = self.Branch3(x)
        x = torch.cat((x0, x1, x2, x3), 1)
        return x

class Block8(nn.Module):
    def __init__(self, scale = 1.0, needReLU = True):
        super().__init__()
        self.Scale = scale
        self.NeedReLU = needReLU
        self.Branch0 = BasicConvolution2D(2080, 192, kernelSize = 1, stride = 1)
        self.Branch1 = nn.Sequential(
            BasicConvolution2D(2080, 192, kernelSize = 1, stride = 1),
            BasicConvolution2D(192, 224, kernelSize = (1, 3), stride = 1, padding = (0, 1)),
            BasicConvolution2D(224, 256, kernelSize = (3, 1), stride = 1, padding = (1, 0)))
        self.Convolution = nn.Conv2d(448, 2080, kernel_size = 1, stride = 1)
        if self.NeedReLU:
            self.ReLU = nn.ReLU(inplace = False)

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.Convolution(out)
        out = out * self.Scale + x
        if self.NeedReLU:
            out = self.ReLU(out)
        return out

class InceptionResNetV2(nn.Module):
    def __init__(self, numOfClasses):
        super().__init__()
        self.Convolution1A = BasicConvolution2D(1, 32, kernelSize = 3, stride = 2)#1,32
        self.Convolution2A = BasicConvolution2D(32, 32, kernelSize = 3, stride = 1)
        self.Convolution2B = BasicConvolution2D(32, 64, kernelSize = 3, stride = 1, padding = 1)
        self.MaxPooling3A = nn.MaxPool2d(3, stride = 2)
        self.Convolution3B = BasicConvolution2D(64, 80, kernelSize = 1, stride = 1)
        self.Convolution4A = BasicConvolution2D(80, 192, kernelSize = 1, stride = 1)
        self.MaxPooling5A = nn.MaxPool2d(3, stride = 2)
        self.Mixed5B = Mixed5B()
        self.Repeat0 = nn.Sequential(
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17))
        self.Mixed6A = Mixed6A()
        self.Repeat1 = nn.Sequential(
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10))
        self.Mixed7A = Mixed7A()
        self.Repeat2 = nn.Sequential(
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20))
        self.Block8 = Block8(needReLU = False)
        self.Convolution7B = BasicConvolution2D(2080, 1536, kernelSize = 1, stride = 1)
        self.AveragePooling1A = nn.AvgPool2d(5, count_include_pad=False)
        self.LastLinear = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, numOfClasses))
        self.fc_CDFI_f = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2)) 
        self.LastLinear1 = nn.Sequential(
            nn.Linear(1536, 512), 
            nn.ReLU(),
            nn.Linear(512, numOfClasses)) 
        self.LastLinear_relu = nn.Sequential(
            nn.Linear(1536, 1536),  
            nn.ReLU(),
            nn.Linear(1536, 1024))


    def forward(self, x,x2,ratio):
        x_f_1 = self.Convolution1A(x)
        x_f_2= self.Convolution2A(x_f_1)
        x_f_3 = self.Convolution2B(x_f_2)
        x_f_4 = self.MaxPooling3A(x_f_3)
        x_f_5 = self.Convolution3B(x_f_4)
        x_f_6 = self.Convolution4A(x_f_5)
        x = self.MaxPooling5A(x_f_6)
        x = self.Mixed5B(x)
        x= self.Repeat0(x)
        x_f_7 = self.Mixed6A(x)
        x = self.Repeat1(x_f_7)
        x_f_8 = self.Mixed7A(x)
        x = self.Repeat2(x_f_8)
        x = self.Block8(x)
        x_f_9 = self.Convolution7B(x)
        x = self.AveragePooling1A(x_f_9)
        x = x.view(x.size(0), -1)
        if ratio ==0:
            x = self.LastLinear1(x)
            return x
        elif ratio==1:
            x2 = x2.view(x2.size(0), -1)
            x_ori = self.LastLinear1(x)
            x2 = self.fc_CDFI_f(x2)#CDFI
            x = torch.cat((x_ori, x2), 1)
            x = self.LastLinear(x)
            return x

        else:#ratio==3，代表是子模型嵌套，InceptionResNetV2_us_cdfi调用
            x = self.LastLinear_relu(x)
            return x


class DMFLNN(nn.Module):
        def __init__(self, numOfClasses):
            super().__init__()
            self.model1 = InceptionResNetV2(2)
            self.model1.Convolution1A = BasicConvolution2D(1, 32, kernelSize=3, stride=2)  # 1,32
            self.model2 = InceptionResNetV2(2)
            self.model2.Convolution1A = BasicConvolution2D(4, 32, kernelSize=3, stride=2)  # 3, 32,
            self.LastLinear_all = nn.Sequential(
                nn.Linear(1024*2, 512),  
                nn.ReLU(),
                nn.Linear(512, numOfClasses))
            self.fc_US = nn.Sequential(
                nn.Linear(1024+689, 512),
                nn.ReLU(),
                nn.Linear(512, 2))#128
            self.fc_C = nn.Sequential(
                nn.Linear(1024+2, 512),
                nn.ReLU(),
                nn.Linear(512, 2))#4
            self.fc_3 = nn.Sequential(
                nn.Linear(4, 4),
                nn.ReLU(),
                nn.Linear(4, numOfClasses))
            
        def forward(self, x, x2, ratio):  # x2为fea
            x_us = x[:,0,:,:].unsqueeze(1)
            x_cdfi = x[:, 1:, :, :]
            f_us = self.model1(x_us,0,3)
            f_cdfi = self.model2(x_cdfi,0,3)
            x2 = x2.view(x2.size(0), -1)
            x2_U=x2[:,:-2]
            x2_C=x2[:,-2:]

            if ratio ==1:#加入fea
                out_2_us = torch.cat((f_us, x2_U), dim=1)
                out_2_us = self.fc_US(out_2_us)
                out_2_cdfi = torch.cat((f_cdfi, x2_C),dim=1)
                out_2_cdfi = self.fc_C(out_2_cdfi)
                out = torch.cat((out_2_us, out_2_cdfi),dim=1)
                x = self.fc_3(out)

            else:#不加fea
                f_map = torch.cat([f_us, f_cdfi], dim=1)
                x = self.LastLinear_all(f_map)

            return x

