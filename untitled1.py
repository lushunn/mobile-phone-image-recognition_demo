# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:33:49 2018

@author: Dell
"""
#爬虫部分
import re
import urllib.request
import os
phonelist={'huawei':8557,'xiaomi':18374,'apple':14026,'meizu':12669,'samsung':15127,'oppo':2032}
for item in phonelist.keys():
    os.mkdir(str(item))
#爬虫
def craw(url,page,brand):
    html1=urllib.request.urlopen(url).read()
    html1=str(html1)
    pat1='<ul class="gl-warp clearfix">.+? <div class="page clearfix">'
    result1=re.compile(pat1).findall(html1)
    result1=result1[0]
    pat2='<img width="220" height="220" data-img="1" data-lazy-img="//.+?\.jpg">'
    imagelist=re.compile(pat2).findall(result1)
    x=1
    for image in imagelist:
        imagename='D:/workfile/pachong/'+str(brand)+'/'+str(page)+str(x)+'.jpg'
        imageurl='http://'+image[60:144]
        try:
            urllib.request.urlretrieve(imageurl,filename=imagename)
        except urllib.error.URLError as e:
            if hasattr(e,'code'):
                x+=1
            if hasattr(e,'reason'):
                x+=1
        x+=1

for brand in list(phonelist.keys()):
    for i in range(1,51):
        url='https://list.jd.com/list.html?cat=9987,653,655&ev=exbrand_'+str(phonelist[str(brand)])+'&page='+str(i)+'&sort=sort_rank_asc&trans=1&JL=6_0_0#J_main'
        craw(url,i,brand)

import Image

#图片压缩批处理  
def compressImage(srcPath,dstPath):  
    for filename in os.listdir(srcPath):  
        #如果不存在目的目录则创建一个，保持层级结构
        if not os.path.exists(dstPath):
                os.makedirs(dstPath)        

        #拼接完整的文件或文件夹路径
        srcFile=os.path.join(srcPath,filename)
        dstFile=os.path.join(dstPath,filename)
        print (srcFile)
        print (dstFile)

        #如果是文件就处理
        if os.path.isfile(srcFile):     
            #打开原图片缩小后保存，可以用if srcFile.endswith(".jpg")或者split，splitext等函数等针对特定文件压缩
            sImg=Image.open(srcFile).convert('RGB')
            w,h=sImg.size  
            print (w,h)
            dImg=sImg.resize((110,110))  #设置压缩尺寸和选项，注意尺寸要用括号
            dImg.save(dstFile) #也可以用srcFile原路径保存,或者更改后缀保存，save这个函数后面可以加压缩编码选项JPEG之类的
            print (dstFile+" 压缩成功")

        #如果是文件夹就递归
        if os.path.isdir(srcFile):
            compressImage(srcFile,dstFile)
            
for key in phonelist.keys():
    compressImage(str(key),str(key)+'1')

import numpy as np
import PIL.Image as Image

#图片to矩阵
def image_to_array(filenames):
    result = np.array([])  # 创建一个空的一维数组
    print("开始将图片转为数组")
    n=len(os.listdir(filenames+'1'))
    for i in os.listdir(filenames+'1'):
        image = Image.open(filenames+'1/'+i)
        r, g, b = image.split()  # rgb通道分离
        r_arr = np.array(r).reshape(12100)
        g_arr = np.array(g).reshape(12100)
        b_arr = np.array(b).reshape(12100)
        # 行拼接
        image_arr = np.concatenate((r_arr, g_arr, b_arr))
        result = np.concatenate((result, image_arr))
    label=list([str(filenames)])
    label=label*n
    return result,label

result= np.array([])
label=list()
for key in phonelist.keys():
    a,b=image_to_array(str(key))
    result = np.concatenate((a, result))
    label=label+b

result = result.reshape((int(len(result)/12100),12100))
label=np.array(label).reshape(2330,1)
data=np.hstack((result,label))
np.savetxt('a.txt',data,fmt='%s',newline='\n')#将结果保存为原始文件

#BP神经网络的单机实现
import numpy as np
import pandas as pd
file=open('a.txt')#导入原始文件
ls=list()
for lines in file:
    line=lines.strip().split(' ')
    ls.append(line)
npy=np.array(ls)
df=pd.DataFrame(npy)
df.columns=list(range(36300))+list('Y')

from sklearn.preprocessing import MinMaxScaler
Newdata=MinMaxScaler().fit_transform(df.iloc[:,:-1])
from sklearn.cross_validation import train_test_split
#Y=df['Y'].map(phonelist)
X=Newdata
Y=df.iloc[:,-1]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=14,
test_size=0.3)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', activation='relu',alpha=1e-4,hidden_layer_sizes=(200,200,200), random_state=1,max_iter=100,verbose=10,learning_rate_init=0.01)
mlp.fit(X_train,Y_train)
pred=mlp.predict(X_test)
corr=0
for i in range(len(pred)):
    if pred[i] == list(Y_test)[i]:
        corr+=1
    else :
        pass
print('正确率:'+str(corr*1.01/len(pred)/1.01))

#Spark上进行BP神经网络建模
from pyspark.sql import Row
from pyspark.ml.feature import Normalizer  
lines = sc.textFile("hdfs:///lushun/a.txt")
parts = lines.map(lambda l: l.split(" "))
df = parts.map(lambda p: Row(features=p[:-1], labe1=int(p[-1])))
df = spark.createDataFrame(df)
df.createOrReplaceTempView("df")
normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)  
l1NormData = normalizer.transform(df)
l1NormData = spark.sql("SELECT labe1,normFeatures FROM l1NormData")  
l1NormData.show()       
from pyspark.ml.classification import MultilayerPerceptronClassifier  
from pyspark.ml.evaluation import MulticlassClassificationEvaluator 
splits = lInfNormData.randomSplit([0.7, 0.3]) 
train = splits[0]  
test = splits[1]
layers = [36300, 200, 200, 6] 
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers,seed=1234)
model = trainer.fit(train)  
# compute accuracy on the test set  
result = model.transform(test)  
predictionAndLabels = result.select("prediction", "label")  
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")  
print("Accuracy: " + str(evaluator.evaluate(predictionAndLabels)))  