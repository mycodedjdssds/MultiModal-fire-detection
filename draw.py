import matplotlib.pyplot as plt
import numpy as np



plt.figure(figsize=(10, 10))  # 窗口大小可以自己设置
fo = open("/home/whj/zxm/C-MHAD-PytorchSolution-master/GeCombine/loss", "r")
a1= fo.read().splitlines()
fo.close()
y1=list(map(float,a1))
x1 = list(range(0,len(y1)))
plt.plot(x1, y1)#label对于的是legend显示的信息名
plt.grid()#显示网格
#plt_title = 'BATCH_SIZE = 48; LEARNING_RATE:0.001'
#plt.title(plt_title)#标题名
#plt.xlabel('per 400 times')#横坐标名
#plt.ylabel('LOSS')#纵坐标名
plt.legend()#显示曲线信息

plt.savefig("test.jpg")#当前路径下保存图片名字
plt.show()



