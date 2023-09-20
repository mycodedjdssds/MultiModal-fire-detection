from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import CMHADDataset
from torch.utils.data import DataLoader
import os
import math
import torch.nn as nn

def timedelta_string(timedelta):
    totalSeconds = int(timedelta.total_seconds())
    hours, remainder = divmod(totalSeconds,60*60)
    minutes, seconds = divmod(remainder,60)
    return "{} hrs, {} mins, {} secs".format(hours, minutes, seconds)

def output_iteration(i, time, totalitems):
    #os.system('clear')

    avgBatchTime = time / (i+1)
    estTime = avgBatchTime * (totalitems - i)

    print("Iteration: {}\nElapsed Time: {} \nEstimated Time Remaining: {}".format(i, timedelta_string(time), timedelta_string(estTime)))

class Trainer():
    def __init__(self, options):
        self.trainingdataset = CMHADDataset(options["training"]["dataset"], "train")
        self.trainingdataloader = DataLoader(
                                    self.trainingdataset,
                                    batch_size=options["input"]["batchsize"],
                                    shuffle=options["input"]["shuffle"],
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=True
                                )
        self.usecudnn = options["general"]["usecudnn"]

        self.batchsize = options["input"]["batchsize"]

        self.statsfrequency = options["training"]["statsfrequency"]

        self.gpuid = options["general"]["gpuid"]

        self.learningrate = options["training"]["learningrate"]

        self.modelType = options["training"]["learningrate"]

        self.weightdecay = options["training"]["weightdecay"]
        self.momentum = options["training"]["momentum"]
        self.modelsavepath = options["general"]["modelsavepath"]

    def learningRate(self, epoch):
        decay = math.floor((epoch - 1) / 2)
        return self.learningrate * pow(0.5, decay)

    def epoch(self, model, epoch):
        #set up the loss function.
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(
                        model.parameters(),
                        lr = self.learningRate(epoch),
                        momentum = self.learningrate,
                        weight_decay = self.weightdecay)

        #transfer the model to the GPU.
        if(self.usecudnn):
            criterion = criterion.cuda(self.gpuid)

        startTime = datetime.now()
        print("Starting training...")
        for i_batch, sample_batched in enumerate(self.trainingdataloader):

            optimizer.zero_grad()
            input = Variable(sample_batched['temporalvolume'])
            labels = Variable(sample_batched['label'])
            a =sample_batched['strii']


            mylist = []
            for i in range(16):
                mylist.append([])  # 在空列表中再添加一个空列表
                mylist[i].append(a[i].split(','))  # 为内层列表添加元素
            #a = a.astype(float)



            if(self.usecudnn):
                input = input.cuda(self.gpuid)
                labels = labels.cuda(self.gpuid)


            outputs = model(input)
            outputs = nn.Softmax(dim=1)(outputs)
            print(outputs)
            outputslist = outputs.data

            print(type(outputslist[0][0]))
            for index in range(16):
                if ((float(mylist[index][0][0]) - 1200) / 1000 )< 0:
                    mylist1=0
                else:
                    mylist1=(float(mylist[index][0][0]) - 1200) / 1000
                if ((float(mylist[index][0][1]) - 1500) / 1000) < 0:
                    mylist2 = 0
                else:
                    mylist2 = (float(mylist[index][0][1]) - 1500) / 1000

                outputslist[index][0] = outputslist[index][0] + math.log10((float(mylist[index][0][5]) - 1) / 100 + 1)+math.log10(mylist1 + 1)+math.log10(mylist2 + 1)
                                        #(float(mylist[index][0][0]) - 1200) / 5000+ (float(mylist[index][0][1]) -1500) / 5000
                if(outputslist[index][0]<0):
                    outputslist[index][0] = 0.000001
                outputslist[index][0] = outputslist[index][0]/(outputslist[index][0]+outputslist[index][1])
                outputslist[index][1] = outputslist[index][1]/(outputslist[index][0]+outputslist[index][1])
            #outputsit = torch.tensor(outputs)
            #if (self.usecudnn):
            #    outputsit = outputsit.cuda(self.gpuid)
            #outputsit.requires_grad_()
            #outputsit = nn.Softmax(dim=1)(outputsit)

            #for index in range(8):
            #    outputs.inplace = False
            #    outputs[index][0] = outputsit[index][0]
            #    outputs.inplace = False
            #    outputs[index][1] = outputsit[index][1]
            #    outputs.inplace = False
            print(outputs)
            outputs = torch.log(outputs)
            print(outputs, labels.squeeze(1))
            loss = criterion(outputs, labels.squeeze(1))
            print(loss)
            loss.backward()
            optimizer.step()
            sampleNumber = i_batch * self.batchsize
            fo = open("/home/whj/zxm/C-MHAD-PytorchSolution-master/GeVideocode（loss）/loss", "a")
            list = str(100 * loss.tolist())
            fo.write(list + "\n")
            fo.close()
            if(sampleNumber % self.statsfrequency == 0):
                currentTime = datetime.now()
                output_iteration(sampleNumber, currentTime - startTime, len(self.trainingdataset))

        print("Epoch completed, saving state...")
        torch.save(model.state_dict(), self.modelsavepath)
