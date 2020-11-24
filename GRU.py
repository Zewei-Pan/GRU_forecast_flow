import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dropout, Dense, GRU
import matplotlib.pyplot as plt
import os


#-------------------数据导入---------------------#
def input_data():
    #读入历史流量数据
    f4=open('q.csv')
    df4=pd.read_csv(f4)
    data4=df4.iloc[552:,1:].values
    
    f4.close()

    '''
    #滑动平滑滤波
    data44=data4
    l4=len(data44)
    data4=np.zeros((l4,1))
    for i in range(1):#滑动平滑滤波次数
        data4[0]=(3*data44[0]+2*data44[1]+data44[2]-data44[3])/5
        data4[1]=(4*data44[0]+3*data44[1]+2*data44[2]+data44[3])/10
        for j in range(2,l4-2):
            data4[j]=np.mean(data44[j-2:j+3])
        data4[l4-2]=(data44[l4-4]+2*data44[l4-3]+3*data44[l4-2]+4*data44[l4-1])/10
        data4[14-1]=(-data44[l4-4]+1*data44[l4-3]+2*data44[l4-2]+3*data44[l4-1])/5
    '''

    datax1=np.zeros((4952-240,1))
    for j in range(4952-240):
        datax1[j][0]=data4[j][0]
   

    datax2=np.zeros((8768-240,1))
    for j in range(8768-240):
        datax2[j][0]=data4[4955+j][0]
    #print(datax2[0])
    #print(datax2[8767-240])

    datay1=np.zeros((4712,1))
    for j in range(4712):
        datay1[j][0]=data4[240+j][0]

    datay2=np.zeros((8528,1))
    for j in range(8528):
        datay2[j][0]=data4[4955+240+j][0]
    return datax1,datax2,datay1,datay2
            
#-------------------数据导入---------------------#

#-------------------获取训练集---------------------#
def get_train_data(batch_size=120,time_step=240,out_time_step=8,train_begin=0,train_end=4560,dataset=1):
    batch_index=[]
    
    if dataset==1:
        data_train=datax1[train_begin:train_end]
        data_train_y=datay1[train_begin:train_end]
    else:
        data_train=datax2[train_begin:train_end]
        data_train_y=datay2[train_begin:train_end]
    train_x,train_y=[],[] #训练集
    for i in range(0,(len(data_train)-time_step),2):
        x=data_train[i:i+time_step,:]
        y=data_train_y[i:i+out_time_step,0,np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    np.random.seed(116)
    np.random.shuffle(train_x)
    np.random.seed(116)
    np.random.shuffle(train_y)
    
    for i in range(0,(len(data_train)-time_step),2):
        if i%batch_size==0:
            batch_index.append(i)
    batch_index.append(len(data_train)-time_step)
    
    return batch_index,train_x,train_y           
#-------------------获取训练集---------------------#


#--------------需要预测的数据-------------#
def data_to_process():
  
    f4=open('qq.csv')
    df4=pd.read_csv(f4)
    data4=df4.iloc[:,1:].values
    f4.close()
    #print(data4[0])
    '''
    data44=data4
    l4=len(data44)
    data4=np.zeros((l4,1))
    for i in range(1):
        data4[0]=(3*data44[0]+2*data44[1]+data44[2]-data44[3])/5
        data4[1]=(4*data44[0]+3*data44[1]+2*data44[2]+data44[3])/10
        for j in range(2,l4-2):
            data4[j]=np.mean(data44[j-2:j+3])
        data4[l4-2]=(data44[l4-4]+2*data44[l4-3]+3*data44[l4-2]+4*data44[l4-1])/10
        data4[14-1]=(-data44[l4-4]+1*data44[l4-3]+2*data44[l4-2]+3*data44[l4-1])/5
    '''
    dataxx=[]
    dataxx1=np.zeros((240,1))
    for j in range(240):
        dataxx1[j][0]=data4[8+j][0]
    dataxx.append(dataxx1.tolist())
    #print(dataxx1)

    dataxx2=np.zeros((240,1))
    for j in range(240):
        dataxx2[j][0]=data4[256+j][0]
    dataxx.append(dataxx2.tolist())

    dataxx3=np.zeros((240,1))
    for j in range(240):
        dataxx3[j][0]=data4[504+j][0]
    dataxx.append(dataxx3.tolist())

    dataxx4=np.zeros((240,1))
    for j in range(240):
        dataxx4[j][0]=data4[752+j][0]
    dataxx.append(dataxx4.tolist())

    dataxx5=np.zeros((240,1))
    for j in range(240):
        dataxx5[j][0]=data4[1000+j][0]
    dataxx.append(dataxx5.tolist())
   
    return dataxx

#--------------需要预测的数据-------------#

datax1,datax2,datay1,datay2=input_data()
batch_index,x_train,y_train=get_train_data(batch_size=32,time_step=240,out_time_step=8,train_begin=0,train_end=6400,dataset=2)
batch_index1,x_test,y_test=get_train_data(batch_size=32,time_step=240,out_time_step=8,train_begin=6400,train_end=8528,dataset=2)

#--------------预测模型------------#
model=tf.keras.Sequential([
    GRU(240,return_sequences=True),
    Dropout(0.1),
    GRU(100,activation='sigmoid'),
    Dropout(0.05),
    Dense(8,activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss='mean_squared_error')

checkpoint_save_path = "./checkpoint/rnn_stock.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')
'''
#训练用
history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=1,verbose=2,
                    callbacks=[cp_callback])
model.summary()
#--------------参数保存及画图------------#
file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
#--------------参数保存及画图------------#
'''
#--------------预测模型------------#

#--------------预测------------#
dataxx=data_to_process()
#print(dataxx)
predict_all=[]
for i in range(5):
    datax11=[]
    datax11.append(dataxx[i])
    #print(datax11)
    predicty=[]
    for j in range(7):
        
        predict1=model.predict(datax11).reshape((-1))
        #print(predict1)
        dataxxx=datax11[0]
        datax12=np.vstack((dataxxx[8:],predict1.reshape((8,1))))
        #print(datax12)
        datax11=[]
        datax11.append(datax12.tolist())
        
        predicty.extend(predict1.tolist())
    #print(predicty)
    predict_all.append(predicty)
#print(predict_all)

name=[]
for i in range(56):
    name.extend(['Prediction'+str(i+1)])
test=pd.DataFrame(columns=name,data=predict_all)
sa='./submission'+'.csv'
test.to_csv(sa)
#--------------预测------------#




