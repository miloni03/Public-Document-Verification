import main as mn
import easyocr
reader=easyocr.Reader(['en','hi'])
import os
import numpy as np
import pickle
import random
#from google.colab import files
from keras.utils import load_img, img_to_array 

from tensorflow.keras.preprocessing.image import ImageDataGenerator
#def analysis(enter_path,path_of_uploaded_file):
docs='C:\\LOC 5.0\\datasets'
for enter_path in os.listdir(docs):
    base_dir = os.path.join(docs,enter_path) #'C:\\LOC 5.0\\okay\\pan_card_dataset'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'test')

# cat and dog folders for training
    train_cats_dir = os.path.join(train_dir, 'real')
    train_dogs_dir = os.path.join(train_dir, 'fake')

# cat and dog folders for validation
    validation_cats_dir = os.path.join(validation_dir, 'real')
    validation_dogs_dir = os.path.join(validation_dir, 'fake')
    import tensorflow as tf

    model = tf.keras.models.Sequential([
    # since Conv2D is the first layer of the neural network, we should also specify the size of the input
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    # apply pooling
        tf.keras.layers.MaxPooling2D(2,2),
    # and repeat the process
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2), 
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
        tf.keras.layers.MaxPooling2D(2,2),
    # flatten the result to feed it to the dense layer
        tf.keras.layers.Flatten(), 
    # and define 512 neurons for processing the output coming by the previous layers
        tf.keras.layers.Dense(512, activation='relu'), 
    # a single output neuron. The result will be 0 if the image is a cat, 1 if it is a dog
        tf.keras.layers.Dense(1, activation='sigmoid')  
    ])
    import tensorflow as tf

    model = tf.keras.models.Sequential([
    # since Conv2D is the first layer of the neural network, we should also specify the size of the input
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    # apply pooling
        tf.keras.layers.MaxPooling2D(2,2),
    # and repeat the process
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2), 
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
        tf.keras.layers.MaxPooling2D(2,2),
    # flatten the result to feed it to the dense layer
        tf.keras.layers.Flatten(), 
    # and define 512 neurons for processing the output coming by the previous layers
        tf.keras.layers.Dense(512, activation='relu'), 
    # a single output neuron. The result will be 0 if the image is a cat, 1 if it is a dog
        tf.keras.layers.Dense(1, activation='sigmoid')  
    ])
    model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics = ['accuracy'])

# we rescale all our images with the rescale parameter
    train_datagen = ImageDataGenerator(rescale = 1.0/255)
    test_datagen  = ImageDataGenerator(rescale = 1.0/255)

# we use flow_from_directory to create a generator for training
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

# we use flow_from_directory to create a generator for validation
    validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode='binary',
                                                         target_size=(150, 150))
    history = model.fit(
            train_generator, # pass in the training generator
            steps_per_epoch=10,
            epochs=10000,
            validation_data=validation_generator, # pass in the validation generator
            validation_steps=50,
            verbose=2
            )
    folder=''
    folder_path=os.listdir(folder)
    d=random.choice(folder_path)

    img =load_img(os.path.join(d,""), target_size=(150,150))
    x = img_to_array(img)
    x /= 255 
    x = np.expand_dims(x, axis=0)
  # flatten the output
    images = np.vstack([x])
    model.save('model.h5')  # prediction!
    classes = model.predict(images, batch_size=10)
    b=classes[0]
    #pickle.dump(model,open('pickle_of_ML_model.pkl','wb'))
    #ml_model=pickle.load(open('pickle_of_ML_model.pkl','rb'))

path_datasets = 'C:\\LOC 5.0\\datasets'
for j in range(1):
    for i in os.listdir(path_datasets):
        a=((os.path.join(path_datasets,i)),os.path.join(d,""))
        output=reader.readtext(os.path.join(d,""),detail=0)
        print(output)
        integers=['1','2','3','4','5','6','7','8','9','0']
        if output[0]=='भारत सरकार' or output[0]=='मारत सरकार' or output[0]=='मारत' or output[1]=='मारत'or output[1]=='भारत सरकार' or output[1]=='भारत' or output[0]=='भारत' or output[1]=='सरकार'  or output[2]=='सरकार' :
            print("valid")
            if (output[2][1]=='G'or 'g') or (output[3][1]=='G'or'g') or(output[2][1]=='O'or 'o') or (output[3][1]=='O'or'0') or (output[4][1]=='G'or'g') or(output[4][1]=='O'or 'o') or (output[1][1]=='G' or 'g'):
                print("Aadhar card uploaded is real")
                break
            else:
                print("upload again")
        #else:
          #  print("aadhar card not detected")
        else:
            print("This is not aadhar card")
        if output[0]=='आयकर' or output[1]=='आयकर' or output[1]=='विभाग' or output[2]=='विभाग' :
            print("valid")
            if (output[4][1]=='I'or 'i') or (output[3][1]=='I'or'i') or (output[5][1]=='I'or'i') or (output[2][1]=='I'or 'i'):
                if (output[4][-1]=='T'or 't') or (output[3][-1]=='T'or't') or (output[5][-1]=='T'or't') or (output[2][-1]=='T'or 't'):
                    print("TPlease Wait..... ")
                    print("Processing Pan Card")
                    print("valid pan card good boy")
                else:
                    print("invalid1")
            else:
                    print("invalid2")
        else:
            print("Neither Aadhar nor pan")
    
