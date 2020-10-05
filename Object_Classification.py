import cv2,os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend as k
k.set_image_dim_ordering('th')
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Flatten,Activation
from keras.layers import MaxPooling2D,Convolution2D
from keras.optimizers import SGD,Adam,RMSprop
datadir='F:\Dataset(AI,ML)\hdataset\prain'
categories=['bike','cars','none','person']
trainingdata=[]
for category in categories:
    path= os.path.join(datadir,category)
    #class_num=categories.index(category)
    for img in os.listdir(path):
        try:
            img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            img_array_resize=cv2.resize(img_array,(128,128))
            trainingdata.append(img_array_resize)
        except Exception as e:
            pass


img_data=np.array(trainingdata)
img_data=img_data.astype('float32')
img_data=img_data/255
print(img_data.shape)
num_channels=1

if num_channels==1:
    if k.image_dim_ordering()=='th':
        img_data=np.expand_dims(img_data,axis=1)
        print(img_data.shape)
    else:
        img_data=np.expands_dims(img_data,axis=4)
        print(img_data.shape)
else:
    if k.image_dim_ordering()=='th':
        img_data=np.rollaxis(img_data,3,1)
        print(img_data.shape)

num_classes=4
num_of_samples=img_data.shape[0]
labels=np.ones((num_of_samples),dtype='int64')
labels[0:338]=0
labels[339:737]=1
labels[738:1118]=2
labels[1119:]=3
Y=np_utils.to_categorical(labels,num_classes)
x,y=shuffle(img_data,Y,random_state=2)
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=4)
input_shape=img_data[0].shape
model=Sequential()
model.add(Convolution2D(32,3,3,border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist=model.fit(X_train,Y_train,batch_size=32,epochs=5,verbose=1,validation_data=(X_test,Y_test))
model.summary()
#model.get_config()
#model.layers[0].get_config()
#model.layers[0].input_shape
#model.layers[0].output_shape
#model.layers[0].get_weights()
#np.shape(model.layers[0].get_weights()[0])
#model.layers[0].trainable
#np.shape(a)
num_epoch=5
model.fit(X_train,Y_train,batch_size=32,epochs=5,verbose=1,validation_data=(X_test,Y_test))
from keras import callbacks
filename='model_train_new.csv'
csv_log=callbacks.CSVLogger(filename,separator=',',append=False)
early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0,patience=0,verbose=1,mode=min)
filepath="best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"
checkpoint=callbacks.ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode=min)
callbacks_list=[csv_log,early_stopping,checkpoint]
hist=model.fit(X_train,Y_train,batch_size=32,epochs=5,verbose=1,callbacks=callbacks_list,validation_data=(X_test,Y_test))
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(2)
#plt.figure(1,figsize=(7,5))
#plt.plot(xc,train_loss)
#plt.plot(xc,val_loss)
#plt.xlabel('num of epochs')
#plt.ylabel('loss')
#plt.title('train_loss vs val_loss')
#plt.grid(True)
#plt.legend(['train','val'])
#plt.style.use(['classic'])
#plt.show()
#plt.figure(2,figsize=(7,5))
#plt.plot(xc,train_acc)
#plt.plot(xc,val_acc)
#plt.xlabel('num of epochs')
#plt.ylabel('accuracy')
#plt.title('train_acc vs val_acc')
#plt.grid(True)
#plt.legend(['train','val'],loc=4)
#plt.style.use(['classic'])
#plt.show()
score=model.evaluate(X_test,Y_test,verbose=0)
print('test loss:',score[0])
print('test accuracy:',score[1])
test_image=X_test[0:1]
print(test_image.shape)
print(model.predict(test_image))
print(model.predict_classes(test_image))
print(Y_test[0:1])
test_image=cv2.imread('F:\hdataset\prain\cars\carsgraz_139.bmp')
test_image=cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(128,128))
test_image=np.array(test_image)
test_image=test_image.astype('float32')
test_image=test_image/255
print(test_image.shape)
if num_channels==1:
    if k.image_dim_ordering()=='th':
        test_image=np.expand_dims(test_image,axis=0)
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)
    else:
        test_image = np.expand_dim(test_image,axis=3)
        test_image = np.expand_dim(test_image,axis=0)
        print(test_image.shape)
else:
    if k.image_dim_ordering()=='th':
        test_image=np.rollaxis(test_image,2,0)
        test_image=np.expand_dims(test_image,axis=0)
        print(test_image.shape)
    else:
        test_image=np.expand_dims(test_image,axis=0)
        print(test_image.shape)
print(model.predict(test_image))
print(model.predict_classes(test_image))
from sklearn.metrics import confusion_matrix
Y_pred=model.predict(X_test)
print(Y_pred)
Y_pred=np.argmax(Y_pred,axis=1)
print(Y_pred)
target_names=['class 0(bikes)','class 1(dogs)','class 2(none)','class 3(person)']
print(confusion_matrix(np.argmax(Y_test,axis=1),Y_pred))

