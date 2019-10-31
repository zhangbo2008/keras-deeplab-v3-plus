import tensorflow as tf
from keras.layers import Input

print(tf.__version__) #必须是2.0.0才行.




from matplotlib import pyplot as plt
import cv2 # used for resize. if you dont have it, use anything else
import numpy as np
from model import Deeplabv3

new_deeplab_model = Deeplabv3(weights='cityscapes',backbone='xception',input_shape=(512,512,3), OS=16,classes=19)  #city数据集对应分类是19

img = plt.imread("imgs/image1.jpg")
w, h, _ = img.shape
ratio = 512. / np.max([w,h])
resized = cv2.resize(img,(int(ratio*h),int(ratio*w)))#归一化处理!
resized = resized / 127.5 - 1. #因为要保持原来的长宽比例,所以短边会需要padding操作才行.
pad_x = int(512 - resized.shape[0])
resized2 = np.pad(resized,((0,pad_x),(0,0),(0,0)), mode='constant') #表示填充到尾部
res = new_deeplab_model.predict(np.expand_dims(resized2,0))
#公式:np.argmax(axis=t) 那么最后的结果中就不包含这个t轴.公式要记住
labels = np.argmax(res.squeeze(),-1) #得到512*512 每一个像素属于哪个物体的编号.直接输出就保证了,图片同一个分类对应的颜色是相同的.

plt.imshow(labels[:-pad_x]) #最后填充的不需要输出,裁剪出去.
plt.savefig("output.png")

