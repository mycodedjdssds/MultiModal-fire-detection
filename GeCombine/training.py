from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import CMHADDataset
from torch.utils.data import DataLoader
import os
import math
import torch.nn as nn
import pdb
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
        decay = math.floor((epoch - 1) / 5)
        return self.learningrate * pow(0.5, decay)

    def epoch(self, model, epoch):
        #set up the loss function.
        criterion = nn.CrossEntropyLoss()
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
            #pdb.set_trace()
            print(len(self.trainingdataloader))
            optimizer.zero_grad()
            input_x = Variable(sample_batched['temporalvolume_x'])
            input_y = Variable(sample_batched['temporalvolume_y'])
            labels = Variable(sample_batched['label'])
            #print(labels )
            #foo = open("/home/whj/zxm/C-MHAD-PytorchSolution-master/GeCombine/datadata", "a")
            #list1 = input_x.numpy().tolist()
            #strNums1 = [str(x) for x in list1]
            #list2 = input_x.numpy().tolist()
            #strNums2 = [str(x) for x in list2]
            #foo.write("".join(strNums1) + '下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下下')
            #foo.write("".join(strNums2))
            #foo.close()
            #print(len(list1), len(list2))
            if(self.usecudnn):
                input_x = input_x.cuda(self.gpuid)
                input_y = input_y.cuda(self.gpuid)
                labels = labels.cuda(self.gpuid)

            outputs = model(input_x,input_y)

            loss = criterion(outputs, labels.squeeze(1))
            print(self.learningrate, loss, labels.squeeze(1))
            loss.backward()
            optimizer.step()
            sampleNumber = i_batch * self.batchsize
            fo = open("/home/whj/zxm/C-MHAD-PytorchSolution-master/GeCombine/loss", "a")
            list = str(100 * loss.tolist())
            fo.write(list + "\n")
            fo.close()
            if(sampleNumber % self.statsfrequency == 0):
                currentTime = datetime.now()
                output_iteration(sampleNumber, currentTime - startTime, len(self.trainingdataset))

        print("Epoch completed, saving state...")
        torch.save(model.state_dict(), self.modelsavepath)


