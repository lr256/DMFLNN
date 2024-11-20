import csv
import random
import numpy as np
random.seed(2223)
from openpyxl import load_workbook as lw#读取xlsx
import cv2
import six

class Folds:
    def __init__(self, benignPatients, malignantPatients, numOfFold, dataPath, patients):#,benignPatients1,benignPatients2,malignantPatients1):#,malignantPatients1,malignantPatients2):
        self.BenignPatients = benignPatients
        self.MalignantPatients = malignantPatients
        self.NumOfFold = numOfFold
        self.DataPath = dataPath
        self.Ratio = 0.11
        self.patients = patients

    def LoadSetData(self, benignStart, benignEnd, malignantStart, malignantEnd, dataPercentage = 1.0):
        setDataIndexList = []
        setLabelList = []
        def LoadBenignOrMalignant(start, end, ps):
            numOfPatients = int((end - start) * dataPercentage)
            for i in range(start, start + numOfPatients):
                setDataIndexList.append(ps[i][0])#把index和label区分
                setLabelList.append(ps[i][1:])
        LoadBenignOrMalignant(int(benignStart), int(benignEnd), self.BenignPatients)
        LoadBenignOrMalignant(int(malignantStart), int(malignantEnd), self.MalignantPatients)

        return np.array(setDataIndexList), np.array(setLabelList)

    def NextFold(self, trainDataPercentage = 1.0):

        fold = {}
        whole = {}
        random.shuffle(self.BenignPatients)  # 随机打乱顺序
        random.shuffle(self.MalignantPatients)  # 随机打乱顺序
        random.shuffle(self.patients)
        whole["DataPath"] = self.DataPath
        whole["ValidationSetDataIndex"], \
        whole["ValidationSetLabel"] = \
            self.LoadSetData(0, round(len(self.BenignPatients) * self.Ratio), \
                             0, round(len(self.MalignantPatients) * self.Ratio))
        fold["DataPath"] = self.DataPath
        fold["TrainSetDataIndex"], \
        fold["TrainSetLabel"] = \
            self.LoadSetData(round(len(self.BenignPatients) * self.Ratio), len(self.BenignPatients), \
                             round(len(self.MalignantPatients) * self.Ratio), len(self.MalignantPatients), \
                             dataPercentage=trainDataPercentage)

        self.BenignPatients = self.BenignPatients[
                              round(len(self.BenignPatients) * self.Ratio): len(self.BenignPatients)] + \
                              self.BenignPatients[0: round(len(self.BenignPatients) * self.Ratio)]
        self.MalignantPatients = self.MalignantPatients[
                                 round(len(self.MalignantPatients) * self.Ratio): len(self.MalignantPatients)] + \
                                 self.MalignantPatients[0: round(len(self.MalignantPatients) * self.Ratio)]
        return fold,whole


    def GetWholeAsVal(self):
        whole = {}
        random.shuffle(self.patients)
        whole["DataPath"] = self.DataPath
        whole["ValidationSetDataIndex"], \
        whole["ValidationSetLabel"] = \
            self.LoadBenignOrMalignant(0, len(self.patients))
        return whole
    def GetWholeAsTest(self):
        whole = {}

        whole["DataPath"] = self.DataPath
        whole["TestSetDataIndex"], \
        whole["TestSetLabel"] = \
            self.LoadBenignOrMalignant(0, len(self.patients))
        return whole


def ReadFolds(paths):
    dataPath, infoPath = paths
    patients = []
    patients_pra = []

    sheet = lw(infoPath).worksheets[0]
    pass_title = False

    i=1
    for row in sheet.values:
        tmp_1 = []

        if not pass_title:
            pass_title = True
            continue
        dataIndex = str(row[0])  # .split(".")[0]
        label = int(row[1])  # 分类
        ratio_chang_duan1 = float(row[6])
        ratio_chang_duan2 = float(row[7])
        ratio_chang_duan3 = float(row[8])
        ratio_chang_duan4 = float(row[9])
        ratio_chang_duan5 = float(row[10])


        for j in range(32,699):
            tmp_1_0 = float(row[j])
            tmp_1.append(tmp_1_0)
        tmp_1= min_max_normalize(tmp_1)

        xingtais1 = float(row[18])
        xingtais2 = float(row[19])
        xingtais3 = float(row[20])
        xingtais4 = float(row[21])
        xingtais5 = float(row[22])
        xingtais6 = float(row[23])
        xingtais7 = float(row[24])
        xingtais8 = float(row[25])
        xingtais9 = float(row[26])
        xingtais10 = float(row[27])
        xingtais11 = float(row[28])
        xingtais12 = float(row[30])  # 血流特征1
        xingtais13 = float(row[31])  # 血流特征2

        patients.append((dataIndex,
                         label,
                         tmp_1[0], tmp_1[1], tmp_1[2], tmp_1[3], tmp_1[4], tmp_1[5], tmp_1[6], tmp_1[7], tmp_1[8],
                         tmp_1[9], tmp_1[10],
                         tmp_1[11], tmp_1[12], tmp_1[13], tmp_1[14], tmp_1[15], tmp_1[16], tmp_1[17], tmp_1[18],
                         tmp_1[19], tmp_1[20],
                         tmp_1[21], tmp_1[22], tmp_1[23], tmp_1[24], tmp_1[25], tmp_1[26], tmp_1[27], tmp_1[28],
                         tmp_1[29], tmp_1[30],
                         tmp_1[31], tmp_1[32], tmp_1[33], tmp_1[34], tmp_1[35], tmp_1[36], tmp_1[37], tmp_1[38],
                         tmp_1[39], tmp_1[40],
                         tmp_1[41], tmp_1[42], tmp_1[43], tmp_1[44], tmp_1[45], tmp_1[46], tmp_1[47], tmp_1[48],
                         tmp_1[49], tmp_1[50],
                         tmp_1[51], tmp_1[52], tmp_1[53], tmp_1[54], tmp_1[55], tmp_1[56], tmp_1[57], tmp_1[58],
                         tmp_1[59], tmp_1[60],
                         tmp_1[61], tmp_1[62], tmp_1[63], tmp_1[64], tmp_1[65], tmp_1[66], tmp_1[67], tmp_1[68],
                         tmp_1[69], tmp_1[70],
                         tmp_1[71], tmp_1[72], tmp_1[73], tmp_1[74], tmp_1[75], tmp_1[76], tmp_1[77], tmp_1[78],
                         tmp_1[79], tmp_1[80],
                         tmp_1[81], tmp_1[82], tmp_1[83], tmp_1[84], tmp_1[85], tmp_1[86], tmp_1[87], tmp_1[88],
                         tmp_1[89], tmp_1[90],
                         tmp_1[91], tmp_1[92], tmp_1[93], tmp_1[94], tmp_1[95], tmp_1[96], tmp_1[97], tmp_1[98],
                         tmp_1[99], tmp_1[100],
                         tmp_1[100], tmp_1[101], tmp_1[102], tmp_1[103], tmp_1[104], tmp_1[105], tmp_1[106], tmp_1[107],
                         tmp_1[108], tmp_1[109], tmp_1[110],
                         tmp_1[111], tmp_1[112], tmp_1[113], tmp_1[114], tmp_1[115], tmp_1[116], tmp_1[117], tmp_1[118],
                         tmp_1[119], tmp_1[120],
                         tmp_1[121], tmp_1[122], tmp_1[123], tmp_1[124], tmp_1[125], tmp_1[126], tmp_1[127], tmp_1[128],
                         tmp_1[129], tmp_1[130],
                         tmp_1[131], tmp_1[132], tmp_1[133], tmp_1[134], tmp_1[135], tmp_1[136], tmp_1[137], tmp_1[138],
                         tmp_1[139], tmp_1[140],
                         tmp_1[141], tmp_1[142], tmp_1[143], tmp_1[144], tmp_1[145], tmp_1[146], tmp_1[147], tmp_1[148],
                         tmp_1[149], tmp_1[150],
                         tmp_1[151], tmp_1[152], tmp_1[153], tmp_1[154], tmp_1[155], tmp_1[156], tmp_1[157], tmp_1[158],
                         tmp_1[159], tmp_1[160],
                         tmp_1[161], tmp_1[162], tmp_1[163], tmp_1[164], tmp_1[165], tmp_1[166], tmp_1[167], tmp_1[168],
                         tmp_1[169], tmp_1[170],
                         tmp_1[171], tmp_1[172], tmp_1[173], tmp_1[174], tmp_1[175], tmp_1[176], tmp_1[177], tmp_1[178],
                         tmp_1[179], tmp_1[180],
                         tmp_1[181], tmp_1[182], tmp_1[183], tmp_1[184], tmp_1[185], tmp_1[186], tmp_1[187], tmp_1[188],
                         tmp_1[189], tmp_1[190],
                         tmp_1[191], tmp_1[192], tmp_1[193], tmp_1[194], tmp_1[195], tmp_1[196], tmp_1[197], tmp_1[198],
                         tmp_1[199], tmp_1[200],
                         tmp_1[200], tmp_1[201], tmp_1[202], tmp_1[203], tmp_1[204], tmp_1[205], tmp_1[206], tmp_1[207],
                         tmp_1[208], tmp_1[209], tmp_1[210],
                         tmp_1[211], tmp_1[212], tmp_1[213], tmp_1[214], tmp_1[215], tmp_1[216], tmp_1[217], tmp_1[218],
                         tmp_1[219], tmp_1[220],
                         tmp_1[221], tmp_1[222], tmp_1[223], tmp_1[224], tmp_1[225], tmp_1[226], tmp_1[227], tmp_1[228],
                         tmp_1[229], tmp_1[230],
                         tmp_1[231], tmp_1[232], tmp_1[233], tmp_1[234], tmp_1[235], tmp_1[236], tmp_1[237], tmp_1[238],
                         tmp_1[239], tmp_1[240],
                         tmp_1[241], tmp_1[242], tmp_1[243], tmp_1[244], tmp_1[245], tmp_1[246], tmp_1[247], tmp_1[248],
                         tmp_1[249], tmp_1[250],
                         tmp_1[251], tmp_1[252], tmp_1[253], tmp_1[254], tmp_1[255], tmp_1[256], tmp_1[257], tmp_1[258],
                         tmp_1[259], tmp_1[260],
                         tmp_1[261], tmp_1[262], tmp_1[263], tmp_1[264], tmp_1[265], tmp_1[266], tmp_1[267], tmp_1[268],
                         tmp_1[269], tmp_1[270],
                         tmp_1[271], tmp_1[272], tmp_1[273], tmp_1[274], tmp_1[275], tmp_1[276], tmp_1[277], tmp_1[278],
                         tmp_1[279], tmp_1[280],
                         tmp_1[281], tmp_1[282], tmp_1[283], tmp_1[284], tmp_1[285], tmp_1[286], tmp_1[287], tmp_1[288],
                         tmp_1[289], tmp_1[290],
                         tmp_1[291], tmp_1[292], tmp_1[293], tmp_1[294], tmp_1[295], tmp_1[296], tmp_1[297], tmp_1[298],
                         tmp_1[299], tmp_1[300],
                         tmp_1[300], tmp_1[301], tmp_1[302], tmp_1[303], tmp_1[304], tmp_1[305], tmp_1[306], tmp_1[307],
                         tmp_1[308], tmp_1[309], tmp_1[310],
                         tmp_1[311], tmp_1[312], tmp_1[313], tmp_1[314], tmp_1[315], tmp_1[316], tmp_1[317], tmp_1[318],
                         tmp_1[319], tmp_1[320],
                         tmp_1[321], tmp_1[322], tmp_1[323], tmp_1[324], tmp_1[325], tmp_1[326], tmp_1[327], tmp_1[328],
                         tmp_1[329], tmp_1[330],
                         tmp_1[331], tmp_1[332], tmp_1[333], tmp_1[334], tmp_1[335], tmp_1[336], tmp_1[337], tmp_1[338],
                         tmp_1[339], tmp_1[340],
                         tmp_1[341], tmp_1[342], tmp_1[343], tmp_1[344], tmp_1[345], tmp_1[346], tmp_1[347], tmp_1[348],
                         tmp_1[349], tmp_1[350],
                         tmp_1[351], tmp_1[352], tmp_1[353], tmp_1[354], tmp_1[355], tmp_1[356], tmp_1[357], tmp_1[358],
                         tmp_1[359], tmp_1[360],
                         tmp_1[361], tmp_1[362], tmp_1[363], tmp_1[364], tmp_1[365], tmp_1[366], tmp_1[367], tmp_1[368],
                         tmp_1[369], tmp_1[370],
                         tmp_1[371], tmp_1[372], tmp_1[373], tmp_1[374], tmp_1[375], tmp_1[376], tmp_1[377], tmp_1[378],
                         tmp_1[379], tmp_1[380],
                         tmp_1[381], tmp_1[382], tmp_1[383], tmp_1[384], tmp_1[385], tmp_1[386], tmp_1[387], tmp_1[388],
                         tmp_1[389], tmp_1[390],
                         tmp_1[391], tmp_1[392], tmp_1[393], tmp_1[394], tmp_1[395], tmp_1[396], tmp_1[397], tmp_1[398],
                         tmp_1[399], tmp_1[400],
                         tmp_1[400], tmp_1[401], tmp_1[402], tmp_1[403], tmp_1[404], tmp_1[405], tmp_1[406], tmp_1[407],
                         tmp_1[408], tmp_1[409], tmp_1[410],
                         tmp_1[411], tmp_1[412], tmp_1[413], tmp_1[414], tmp_1[415], tmp_1[416], tmp_1[417], tmp_1[418],
                         tmp_1[419], tmp_1[420],
                         tmp_1[421], tmp_1[422], tmp_1[423], tmp_1[424], tmp_1[425], tmp_1[426], tmp_1[427], tmp_1[428],
                         tmp_1[429], tmp_1[430],
                         tmp_1[431], tmp_1[432], tmp_1[433], tmp_1[434], tmp_1[435], tmp_1[436], tmp_1[437], tmp_1[438],
                         tmp_1[439], tmp_1[440],
                         tmp_1[441], tmp_1[442], tmp_1[443], tmp_1[444], tmp_1[445], tmp_1[446], tmp_1[447], tmp_1[448],
                         tmp_1[449], tmp_1[450],
                         tmp_1[451], tmp_1[452], tmp_1[453], tmp_1[454], tmp_1[455], tmp_1[456], tmp_1[457], tmp_1[458],
                         tmp_1[459], tmp_1[460],
                         tmp_1[461], tmp_1[462], tmp_1[463], tmp_1[464], tmp_1[465], tmp_1[466], tmp_1[467], tmp_1[468],
                         tmp_1[469], tmp_1[470],
                         tmp_1[471], tmp_1[472], tmp_1[473], tmp_1[474], tmp_1[475], tmp_1[476], tmp_1[477], tmp_1[478],
                         tmp_1[479], tmp_1[480],
                         tmp_1[481], tmp_1[482], tmp_1[483], tmp_1[484], tmp_1[485], tmp_1[486], tmp_1[487], tmp_1[488],
                         tmp_1[489], tmp_1[490],
                         tmp_1[491], tmp_1[492], tmp_1[493], tmp_1[494], tmp_1[495], tmp_1[496], tmp_1[497], tmp_1[498],
                         tmp_1[499], tmp_1[500],
                         tmp_1[500], tmp_1[501], tmp_1[502], tmp_1[503], tmp_1[504], tmp_1[505], tmp_1[506], tmp_1[507],
                         tmp_1[508], tmp_1[509], tmp_1[510],
                         tmp_1[511], tmp_1[512], tmp_1[513], tmp_1[514], tmp_1[515], tmp_1[516], tmp_1[517], tmp_1[518],
                         tmp_1[519], tmp_1[520],
                         tmp_1[521], tmp_1[522], tmp_1[523], tmp_1[524], tmp_1[525], tmp_1[526], tmp_1[527], tmp_1[528],
                         tmp_1[529], tmp_1[530],
                         tmp_1[531], tmp_1[532], tmp_1[533], tmp_1[534], tmp_1[535], tmp_1[536], tmp_1[537], tmp_1[538],
                         tmp_1[539], tmp_1[540],
                         tmp_1[541], tmp_1[542], tmp_1[543], tmp_1[544], tmp_1[545], tmp_1[546], tmp_1[547], tmp_1[548],
                         tmp_1[549], tmp_1[550],
                         tmp_1[551], tmp_1[552], tmp_1[553], tmp_1[554], tmp_1[555], tmp_1[556], tmp_1[557], tmp_1[558],
                         tmp_1[559], tmp_1[560],
                         tmp_1[561], tmp_1[562], tmp_1[563], tmp_1[564], tmp_1[565], tmp_1[566], tmp_1[567], tmp_1[568],
                         tmp_1[569], tmp_1[570],
                         tmp_1[571], tmp_1[572], tmp_1[573], tmp_1[574], tmp_1[575], tmp_1[576], tmp_1[577], tmp_1[578],
                         tmp_1[579], tmp_1[580],
                         tmp_1[581], tmp_1[582], tmp_1[583], tmp_1[584], tmp_1[585], tmp_1[586], tmp_1[587], tmp_1[588],
                         tmp_1[589], tmp_1[590],
                         tmp_1[591], tmp_1[592], tmp_1[593], tmp_1[594], tmp_1[595], tmp_1[596], tmp_1[597], tmp_1[598],
                         tmp_1[599], tmp_1[600],
                         tmp_1[600], tmp_1[601], tmp_1[602], tmp_1[603], tmp_1[604], tmp_1[605], tmp_1[606], tmp_1[607],
                         tmp_1[608], tmp_1[609], tmp_1[610],
                         tmp_1[611], tmp_1[612], tmp_1[613], tmp_1[614], tmp_1[615], tmp_1[616], tmp_1[617], tmp_1[618],
                         tmp_1[619], tmp_1[620],
                         tmp_1[621], tmp_1[622], tmp_1[623], tmp_1[624], tmp_1[625], tmp_1[626], tmp_1[627],
                         tmp_1[628], tmp_1[629], tmp_1[630],
                         tmp_1[631], tmp_1[632], tmp_1[633], tmp_1[634], tmp_1[635], tmp_1[636], tmp_1[637], tmp_1[638],
                         tmp_1[639], tmp_1[640],
                         tmp_1[641], tmp_1[642], tmp_1[643], tmp_1[644], tmp_1[645], tmp_1[646], tmp_1[647], tmp_1[648],
                         tmp_1[649], tmp_1[650],
                         tmp_1[651], tmp_1[652], tmp_1[653], tmp_1[654], tmp_1[655], tmp_1[656], tmp_1[657], tmp_1[658],
                         tmp_1[659], tmp_1[660],
                         tmp_1[661], tmp_1[662], tmp_1[663], tmp_1[664], tmp_1[665], tmp_1[666],
                         ratio_chang_duan1, ratio_chang_duan2, ratio_chang_duan3, ratio_chang_duan4, ratio_chang_duan5,
                         xingtais1, xingtais2, xingtais3, xingtais4, xingtais5,
                         xingtais6, xingtais7, xingtais8, xingtais9, xingtais10, xingtais11, xingtais12, xingtais13
                         ))

    benignPatients = []
    malignantPatients = []

    if patient[1] == 0:#[1] == 0:
        benignPatients.append(patient)
    else:
        malignantPatients.append(patient)

    folds = Folds(benignPatients, malignantPatients, 5, dataPath, patients)
    return folds
