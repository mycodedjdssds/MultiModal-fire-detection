import imageio
imageio.plugins.ffmpeg.download()
import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
import torch
from .statefultransforms import StatefulRandomCrop, StatefulRandomHorizontalFlip
import pandas
import pdb
import numpy as np

def load_video(filename, startframe):
    """Loads the specified video using ffmpeg.

    Args:
        filename (str): The path to the file to load.
            Should be a format that ffmpeg can handle.

    Returns:
        List[FloatTensor]: the frames of the video as a list of 3D tensors
            (channels, width, height)"""

    try:
        vid = imageio.get_reader(filename,  'ffmpeg')
    except:
        print("Error at: "+filename)    
    frames = []
    for i in range(0, 60):
        image = vid.get_data(startframe+i)
        image = functional.to_tensor(image)
        frames.append(image)
    return frames

def bbc(vidframes, augmentation=True):
    """Preprocesses the specified list of frames by center cropping.
    This will only work correctly on videos that are already centered on the
    mouth region, such as LRITW.

    Args:
        vidframes (List[FloatTensor]):  The frames of the video as a list of
            3D tensors (channels, width, height)

    Returns:
        FloatTensor: The video as a temporal volume, represented as a 5D tensor
            (batch, channel, time, width, height)"""

    temporalvolume = torch.FloatTensor(1,60,240,320)

    for i in range(0, 60):
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((240,320)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])(vidframes[i])

        temporalvolume[0][i] = result

    return temporalvolume


def load_inertial(filename, startframe):
    if startframe<0:
        startframe=0
    df = pandas.read_csv(filename)
    df.columns = ['A', 'B', 'C', 'D', 'E', 'F']
    #df = df.iloc[2:,1:]
    df = df.astype(float)
    #print(df)
    #dfmean = df.mean(axis=0)
    #print(dfmean)
    #df = df.div(dfmean,axis=1)
    df = df.to_numpy()
    #df[:, 0] = (df[:, 0] - 800) / 2700
    #df[:, 1] = (df[:, 1] - 1000) / 1800
    #df[:, 2] = (df[:, 2] - 1800) / 2500
    #df[:, 3] = (df[:, 3] - 0) / 4500
    #df[:, 4] = (df[:, 4] - 1800) / 2500
    #df[:, 5] = (df[:, 5] - 0) / 300
    #print(df.shape)
    #acc = pow((pow(df[:,0],2)+pow(df[:,1],2)+pow(df[:,2],2)),0.5)
    #gyr = pow((pow(df[:,3],2)+pow(df[:,4],2)+pow(df[:,5],2)),0.5)
    #acc = np.asarray(acc)
    #gyr = np.asarray(gyr)
    #df = np.concatenate((df, acc[:, np.newaxis],gyr[:, np.newaxis]), axis=1)
    #print(df.shape)
    #print(df)
    #df_normed = (df-df.min(axis=0)) / (df.max(axis=0)-df.min(axis=0))
    #print(df_normed)

    frames = df[startframe:startframe+300:5,:]
    frames = frames[np.newaxis,:]
    #print(frames.shape)
    frames = torch.from_numpy(frames).float()
    #print("########Inertial Load Done#######")
    return frames

