from torch.utils.data import Dataset
from .preprocess import *
import os
import xlrd
import pandas
import pdb
class CMHADDataset(Dataset):
    """BBC Lip Reading dataset."""

    def build_file_list(self, dir, set):
        labels = ['Action1','Action2']
        completeList = []
        subject = 1
        while subject<=1:
            '''修改文件数目'''
            subdir = dir+"/Subject"+str(subject)
            files = os.listdir(subdir)
            LabelPath = xlrd.open_workbook(subdir+"/ActionOfInterestTraSubject"+str(subject)+".xlsx")
            sheet = LabelPath.sheet_by_index(0) 
            #print(str(sheet.nrows)+" rows in Total")
            min = 2
            max = 3
            '''大小指的是每个动作持续时间的长短'''
            for l in range(sheet.nrows-1): 
                val = sheet.cell_value(l+1, 3)-sheet.cell_value(l+1, 2)
                if val < min:
                    min = val
                if val > max:
                    max = val
            print("The Minimum action duration of this Subject is: "+str(min)+" seconds")
            print("The Maximum action duration of this Subject is: "+str(max)+" seconds")
            valvideo = [2]
            testvideo = [18]
            print("Validation Video include:", str(valvideo[0]))

            for m in range(sheet.nrows):
                if m==0:
                    continue
                dirpath = subdir + "/InertialData/inertial_sub"+str(subject)+"_tr"+str(int(sheet.cell_value(m, 0)))+".csv"
                df = pandas.read_csv(dirpath)
                #print(df)
                MissFrames = 5405-len(df.index)
                midtime = sheet.cell_value(m, 3)/2 + sheet.cell_value(m, 2)/2
                '''cell函数是返回某一引用区域的左上角单元格的格式、位置或内容等信息_value返回数值'''
                #midframe = MissFrames+int(50*midtime)  #framerate = 50; starting from 0 fram, indicating 0.00 seconds.   
                if (set == "val") and (sheet.cell_value(m, 0) in valvideo) :
                    print("Creating Vallidation dataset for Action"+ str(int(sheet.cell_value(m, 1))), dirpath)
                    startframe = int(64*midtime) - 150 #150 frames in total
                    endframe = int(64*midtime) + 149
                    startframe, endframe = self.check_overflow(startframe, endframe)
                    entry = (int(sheet.cell_value(m, 1)-1), dirpath, startframe, startframe+299,MissFrames,subject)
                    completeList.append(entry)
                
                elif (set == "train") and (sheet.cell_value(m, 0) not in valvideo) and(sheet.cell_value(m, 0) not in testvideo):
                    print("Creating Training dataset for Action"+ str(int(sheet.cell_value(m, 1))), dirpath)
                    startframe = int(64*midtime) - 180 #200frames in total length, using only 150 frames from 200 as data augmentation
                    endframe = int(64*midtime) + 179
                    startframe, endframe = self.check_overflow(startframe, endframe)
                    for n in range(60):
                        #entry = (int(sheet.cell_value(m, 1)-1), dirpath, startframe+3*n, startframe+3*n+149,MissFrames,subject)
                        entry = (int(sheet.cell_value(m, 1)-1), dirpath, startframe+n, startframe+n+299,MissFrames,subject)
                        completeList.append(entry)
            if set == "test":
                for o in testvideo:
                    startframe = MissFrames
                    dirpath = subdir + "/InertialData/inertial_sub"+str(subject)+"_tr"+str(o)+".csv"
                    print("Creating Testing dataset for", dirpath)
                    while startframe <= 5075:
                        entry = (0, dirpath, startframe, startframe+299, MissFrames,subject)
                        completeList.append(entry)            
                        startframe = startframe + 64
            subject = 1+subject           
        print("Size of data : " + str(len(completeList)))       
        #print(completeList)
        return labels, completeList

    def check_overflow(self, startframe, endframe):
        '''防止溢出'''
        if startframe < 4: #avoid overflow
            endframe =  endframe+4-startframe
            startframe = 4
        elif endframe > 5404:
            startframe = startframe - (endframe-5404)
            endframe = 5404
        return startframe, endframe
           
    def __init__(self, directory, set, augment=True):
        self.label_list, self.file_list = self.build_file_list(directory, set)
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        label, filename, startframe, endframe, MissFrames,subject = self.file_list[idx]
        print(MissFrames)
        Inerframes = load_inertial(filename, startframe-MissFrames)
        sample = {'temporalvolume': Inerframes, 'label': torch.LongTensor([label]), 'MiddleTime':(startframe+150)/60, 'subject':subject, }
        return sample
