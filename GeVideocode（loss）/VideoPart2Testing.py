from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import toml
import math
import cv2
from training import Trainer
from validation import Validator
from datetime import datetime, timedelta
from data import CMHADDataset
from torch.utils.data import DataLoader
import numpy as np
import os, xlrd
print("Loading options...")
with open('options.toml', 'r') as optionsFile:
    options = toml.loads(optionsFile.read())
if(options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
    print("Running cudnn benchmark...")
    torch.backends.cudnn.benchmark = True
flag = 0
#load the model.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, (18,3,3), stride=(1,2,2), padding=(0,1,1))
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, (3,3,3), stride=(1,1,2), padding=(1,1,1))
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(2, stride=2)      
        self.pool2 = nn.MaxPool3d((2,2,2), stride=(1, 2, 2)) 
        self.fc1 = nn.Linear(64 * 2 * 7 * 5, 128)
        self.dense1_bn1 = nn.BatchNorm1d(128)
        #self.fc2 = nn.Linear(512, 64)
        #self.dense1_bn2 = nn.BatchNorm1d(64)
        self.dr1 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, 2)
      
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.pool1(F.relu(self.bn3(self.conv3(x))))
        x = self.pool1(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 64 * 2 * 7 * 5)
        x = F.relu(self.dense1_bn1(self.fc1(x)))
        #pdb.set_trace()
        x = self.dr1(x)
        x = self.fc3(x)
        return x
model= Net()

print(options["general"]["modelsavepath"])
model.load_state_dict(torch.load(options["general"]["modelsavepath"]))
model.eval()
print(model)
#Move the model to the GPU.
if(options["general"]["usecudnn"]):
    model = model.cuda(options["general"]["gpuid"])

#load Testing model.
testdataset = CMHADDataset(options["validation"]["dataset"],"test", False)
testdataloader = DataLoader(
                                    testdataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=True
                                )

#Testing and output to 
print("Starting Testing...")
TimeRestrict = 0
Lastmaxindices = -1
Lastmaxvalues = -1
count = 0
ResultList = []
TP_DetectionOnly = 0
TP_DetectionAndRecognition = 0
PredictPositive = 0
ActualPositive = 0
for i_batch, sample_batched in enumerate(testdataloader):
    #if(i_batch>100):
    #    break
    print(i_batch)
    input = Variable(sample_batched['temporalvolume'])
    labels = sample_batched['label']
    a = sample_batched['strii']
    MiddleTime = sample_batched['MiddleTime']
    subject = sample_batched['subject']
    if(options["general"]["usecudnn"]):
        input = input.cuda(options["general"]["gpuid"])
        labels = labels.cuda(options["general"]["gpuid"])
    print(a)
    mylist = a

    #mylist.append(a[0].split(','))  # 为内层列表添加元素
    #    mylist.append([])  # 在空列表中再添加一个空列表
     #   mylist[i].append(a[i].split(','))  # 为内层列表添加元素
    # a = a.astype(float)
    outputs = model(input)
    outputs = nn.Softmax(dim=1)(outputs)
    print(outputs)
    outputslist = outputs.data

    print(mylist[5])
    if ((float(mylist[0]) - 1200) / 1000) < 0:
        mylist1 = 0
    else:
        mylist1 = (float(mylist[0]) - 1200) / 1000
    if ((float(mylist[1]) - 1500) / 1000) < 0:
        mylist2 = 0
    else:
        mylist2 = (float(mylist[1]) - 1500) / 1000

    outputslist[0][0] = outputslist[0][0] +math.log10((float(mylist[5]) - 1) / 100 + 1) +math.log10(mylist1 + 1)+math.log10(mylist2 + 1)
                        #(float(mylist[5]) - 1) / 1000 +(float(mylist[0]) - 1200) / 10000+ (float(mylist[1]) -1500) / 10000
                        #
                        #math.log10((float(mylist[5]) - 1) / 100 + 1)
    if (outputslist[0][0] < 0):
        outputslist[0][0] = 0.000001
    outputslist[0][0] = outputslist[0][0] / (outputslist[0][0] + outputslist[0][1])
    outputslist[0][1] = outputslist[0][1] / (outputslist[0][0] + outputslist[0][1])
    thedata = outputs.tolist()
    thedata2 = ' '.join(str(k) for k in thedata)
    thedata1 = [str(k) for k in thedata2]
    print(outputs)
    str1 = ''.join(thedata1)
    print(str1)
    if(flag==0):
        fo = open("/home/whj/zxm/pythoweb/static/img/test", "w")
        fo.write(str1)
    if(flag!=0):
        fo = open("/home/whj/zxm/pythoweb/static/img/test", "a")
        fo.write("\n"+str1)
    fo.close()
    flag=1
    maxvalues, maxindices = torch.max(outputs.data, 1)
    ResultList.append([MiddleTime[0].tolist()]+outputs[0].data.cpu().tolist())

subject = 1
while subject<=1:
    subdir = options["validation"]["dataset"]+"/Subject"+str(subject)
    files = os.listdir(subdir)
    LabelPath = xlrd.open_workbook(subdir+"/ActionOfInterestTraSubject"+str(subject)+".xlsx")
    print('zheshiyigeren')
    sheet = LabelPath.sheet_by_index(0) 
    for m in range(sheet.nrows):
        if m==0:
            continue
        if sheet.cell_value(m, 0) == 4:
            ActualPositive = ActualPositive+1
    subject = subject+1

'''
def validate(modelOutput, labels):
    maxvalues, maxindices = torch.max(modelOutput.data, 1)
    count = 0
    for i in range(0, labels.squeeze(1).size(0)):
        if maxindices[i] == labels.squeeze(1)[i]:
            count += 1
    return count
    
validationdataset = CMHADDataset(options["validation"]["dataset"],"val", False)
validationdataloader = DataLoader(
                            validationdataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=options["input"]["numworkers"],
                            drop_last=False
                        )
count = 0

for i_batch, sample_batched in enumerate(validationdataloader):
    input = Variable(sample_batched['temporalvolume'])
    labels = sample_batched['label']
    if(options["general"]["usecudnn"]):
        input = input.cuda(options["general"]["gpuid"])
        labels = labels.cuda(options["general"]["gpuid"])
    outputs = model(input)
    count += validate(outputs, labels)
    print(count)
    accuracy = count / len(validationdataset)
with open(options["testing"]["resultfilelocation"], "a") as outputfile:
    outputfile.write("\ncorrect count: {}, total count: {} accuracy: {}" .format(count, len(validationdataset), accuracy ))
'''