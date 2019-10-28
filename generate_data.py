import cv2
from time import sleep
import os
from tkinter import *
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img
from numpy import array
from keras import regularizers
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
from matplotlib.pyplot import imshow
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import backend as K
import os





import pathlib
from tkinter import *
import tkinter as ttk
import shutil
import os


# define the path





    

def add():
    if not os.path.exists("Dataset"):os.mkdir("Dataset")
    if not os.path.exists("Dataset/training_set"):os.mkdir("Dataset/training_set")
    if not os.path.exists("Dataset/test_set"):os.mkdir("Dataset/test_set")

    dirs=[''+close_window()+'/'+close_window()+'']
    sets={'training_set':10000,'test_set':1000}

    for set_name in sets:
        #print("Taking images for the {}. Press enter when ready. ".format(set_name.upper()))
        ##input()
        if not os.path.exists("Dataset"):os.mkdir("Dataset/{}".format(set_name))
        for dir_name in dirs:
            #print("""\nTaking images for the {} dataset. Press enter whenever ready.
            #Note: Place the gesture to be recorded inside the green rectangle shown in the preview until it automatically disappears.""".format(dir_name))
            ##input()
            #for _ in range(3):
                #print(3-_)
                #sleep(1)
            #print("GO!")
            if not os.path.exists("Dataset/{}/{}".format(set_name,os.path.basename(dir_name))):os.mkdir("Dataset/{}/{}".format(set_name,os.path.basename(dir_name)))
            vc=cv2.VideoCapture(0)
            if vc.isOpened():
                rval,frame= vc.read()
            else:
                rval=False
            index=0
            
            while rval:
                input()
                ##sleep(0.1)
                frame=frame[200:400,300:500]
                ##frame = cv2.resize(frame, (200,200))
                frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY)
                frame=frame.reshape((1,)+frame.shape)
                frame=frame.reshape(frame.shape+(1,))
                cv2.destroyWindow("preview")
                index+=1
                rval, frame = vc.read()
                frame=cv2.flip(frame,1)
                cv2.putText(frame,"Keep your hand insid", (20,50), cv2.FONT_HERSHEY_PLAIN , 1, 255)
                cv2.putText(frame,"Taking images for {} dataset".format(dir_name), (20,80), cv2.FONT_HERSHEY_PLAIN , 1, 255)
                cv2.rectangle(frame,(300,200),(500,400),(0,255,0),1)
                cv2.imshow("Recording", frame)
                cv2.imwrite("Dataset/{}/".format(set_name)+str(dir_name)+"{}.jpg".format(index),frame[200:400,300:500]) #save image
                print("images taken: {}".format(index))
                key = cv2.waitKey(20)
                if key == 27 or index==sets[set_name]: # exit on ESC or when enough images are taken
                    break

            cv2.destroyWindow("Recording")
            vc=None



def train():
            #init the model
     
       
        model= Sequential()

        #add conv layers and pooling layers 
        model.add(Convolution2D(32,3,3, input_shape=(200,200,1),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Convolution2D(32,3,3, input_shape=(200,200,1),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Dropout(0.5)) #to reduce overfitting

        model.add(Flatten())

        #Now two hidden(dense) layers:
        model.add(Dense(output_dim = 150, activation = 'relu',
                        kernel_regularizer=regularizers.l2(0.01)))

        model.add(Dropout(0.5))#again for regularization

        model.add(Dense(output_dim = 150, activation = 'relu',
                        kernel_regularizer=regularizers.l2(0.01)))


        model.add(Dropout(0.5))#last one lol

        model.add(Dense(output_dim = 150, activation = 'relu',
                        kernel_regularizer=regularizers.l2(0.01)))

        #output layer
        
        
        model.add(Dense(output_dim = num(), activation = 'sigmoid'))


        #Now copile it
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


        #Now generate training and test sets from folders

        train_datagen=ImageDataGenerator(
                                           rescale = 1./255,
                                           shear_range = 0.2,
                                           zoom_range = 0.,
                                           horizontal_flip = False
                                         )

        test_datagen=ImageDataGenerator(rescale=1./255)

        training_set=train_datagen.flow_from_directory(r"C:\Users\Obada\Desktop\Hand-Gesture-Recognizer-master\dataset\training_set",
                                                       target_size = (200,200),
                                                       color_mode='grayscale',
                                                       batch_size=10,
                                                       class_mode='categorical')

        test_set=test_datagen.flow_from_directory(r"C:\Users\Obada\Desktop\Hand-Gesture-Recognizer-master\dataset\test_set",
                                                       target_size = (200,200),
                                                       color_mode='grayscale',
                                                       batch_size=10,
                                                       class_mode='categorical')






        #finally, start training
        model.fit_generator(training_set,
                                 samples_per_epoch = 19707,
                                 nb_epoch = 10,
                                 validation_data = test_set,
                                 nb_val_samples = 320)


        #after 10 epochs:
            #training accuracy: 0.9005
            #training loss:     0.4212
            #test set accuracy: 0.8813
            #test set loss:     0.5387

        #saving the weights
        model.save_weights("weights.hdf5",overwrite=True)

        #saving the model itself in json format:
        model_json = model.to_json()
        with open("model.json", "w") as model_file:
            model_file.write(model_json)
        print("Model has been saved.")


        #testing it to a random image from the test set
        img = load_img('Dataset/test_set/five/five26.jpg',target_size=(200,200))
        x=array(img)
        img = cv2.cvtColor( x, cv2.COLOR_RGB2GRAY )
        img=img.reshape((1,)+img.shape)
        img=img.reshape(img.shape+(1,))

        test_datagen = ImageDataGenerator(rescale=1./255)
        m=test_datagen.flow(img,batch_size=1)
        y_pred=model.predict_generator(m,1)


        #save the model schema in a pic
        plot_model(model, to_file='model.png', show_shapes = True)
   
histarray={'Empty':0, 'One':0,'Tow':0,'Zero':0}

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 18:48:04 2017

@author: Yugal
"""




histarray={'Empty':0, 'One':0,'Tow':0,'Zero':0}
def num():
        path=r"C:\Users\Obada\Desktop\Hand-Gesture-Recognizer-master\dataset\training_set"
        os.chdir(path)
        list = os.listdir() # dir is your directory path
        number_files = len(list)
        
        return number_files
def load_model():
    try:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("weights.hdf5")
        print("Model successfully loaded from disk.")
            
            #compile again
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return model
    except:
        print("""Model not found. Please train the CNN by running the script 
    cnn_train.py. Note that the training and test samples should be properly 
    set up in the dataset directory.""")
        return None
        
        
def visualize( img, layer_index=0, filter_index=0 ,all_filters=False ):
        
    act_fun = K.function([model.layers[0].input, K.learning_phase()], 
                                    [model.layers[layer_index].output,])
        
        ##img = load_img('Dataset/test_set/three/70.jpg',target_size=(200,200))
    x=img_to_array(img)
    img = cv2.cvtColor( x, cv2.COLOR_RGB2GRAY )
    img=img.reshape(img.shape+(1,))
    img=img.reshape((1,)+img.shape)
    img = act_fun([img,0])[0]
        
    if all_filters:
        fig=plt.figure(figsize=(7,7))
        filters = len(img[0,0,0,:])
        for i in range(filters):
                plot = fig.add_subplot(6, 6, i+1)
                plot.imshow(img[0,:,:,i],'gray')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
        plt.tight_layout()
    else:
        img = np.rollaxis(img, 3, 1)
        img=img[0][filter_index]
        print(img.shape)
        imshow(img)


def update(histarray2):
    global histarray
    histarray=histarray2


    #realtime:
def realtime():
          #initialize preview
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
        
    if vc.isOpened(): #get the first frame
        rval, frame = vc.read()
            
    else:
        rval = False
        
    classes=["empty","one","tow","zero"]
        
    while rval:
        frame=cv2.flip(frame,1)
        cv2.rectangle(frame,(300,200),(500,400),(0,255,0),1)
        cv2.putText(frame,"Place your hand in the green box.", (50,50), cv2.FONT_HERSHEY_PLAIN , 1, 255)
        cv2.putText(frame,"Press esc to exit.", (50,100), cv2.FONT_HERSHEY_PLAIN , 1, 255)
            
        cv2.imshow("preview", frame)
        frame=frame[200:400,300:500]
            #frame = cv2.resize(frame, (200,200))
        frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY)
        frame=frame.reshape((1,)+frame.shape)
        frame=frame.reshape(frame.shape+(1,))
        test_datagen = ImageDataGenerator(rescale=1./255)
        m=test_datagen.flow(frame,batch_size=1)
        y_pred=model.predict_generator(m,1)
        histarray2={'Empty': y_pred[0][0],  'One': y_pred[0][1],'Three': y_pred[0][2],'Zero': y_pred[0][3]}
        update(histarray2)
        print(classes[list(y_pred[0]).index(y_pred[0].max())])
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
                break
    cv2.destroyWindow("preview")
    vc=None
        

      #loading the model
    
model=load_model()
            #visualize(load_img('Dataset/test_set/zero/zero1.jpg',target_size=(200,200)),filter_index=0,all_filters=True)


#if model is not None:
                
                #ans=str(input("Do you want to plot a realtime histogram as well? (slower) y/n\n"))
                
                #if ans.lower()=='y':
                    #the code for histogram 
                 #   fig = plt.figure()
                  #  ax1 = fig.add_subplot(1, 1, 1)
                    
                   # def animate(i):
                    #    xar= [1, 2, 3, 4 ,5 ,6,7]
                     #   yar = []
                      #  xtitles = ['']
                       # for items in histarray:
                        #    yar.append(histarray[items])
                         #   xtitles.append(items)
                        #
                        #ax1.clear()        
                        #plt.bar(xar,yar, align='center')
                        #plt.xticks(np.arange(8), xtitles)
                        
                    #ani = animation.FuncAnimation(fig, animate, interval=500)
                    #fig.show()

                #threading.Thread(target=realtime).start()
    #realtime()

def close_window():
    global entry
    entry = nete2.get()
    return entry

root=Tk()
print("")
Label= Label(root ,text="Enter the name of a new gesture")
Label.grid(row=0,column=0)
nete2=Entry(root)
nete2.grid(row=1,column=0)
currentDirectory = pathlib.Path(r'C:\Users\Obada\Desktop\Hand-Gesture-Recognizer-master\dataset\training_set')
v=[]
for currentFile in currentDirectory.iterdir():  
    print(currentFile)
    v.append(currentFile)


variable = StringVar(root)
variable.set("Select a gesture you want to delete") # default value

global w
w= OptionMenu(root,variable , *v)
w.grid(row =3)
def printt():
    global m

    m=variable.get()
    print(m)
def delete():
    shutil.rmtree(variable.get())
but3=Button(root,text='delete',fg='red',command=delete)
but3.grid(row=3,column=1)



but=Button(root,text='add',fg='red',command=add)
but.grid(row=1,column=1)
but1=Button(root,text='retrain',fg='red',command=train)
but1.grid(row=4)
but2=Button(root,text='test',fg='red',command=realtime)
but2.grid(row=5)
root.mainloop()

