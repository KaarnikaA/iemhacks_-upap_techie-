
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, GlobalAveragePooling2D 
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
import pickle
img_height,img_width=(224,224)
batch_size=40
train_data_dir=r"E:\Projects\Cancer_prediction\DL\output\train"
valid_data_dir=r"E:\Projects\Cancer_prediction\DL\output\val"
test_data_dir=r"E:\Projects\Cancer_prediction\DL\output\test"
#image generator= for generating agumented image
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,
                               shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                validation_split=.4)
train_generator=train_datagen.flow_from_directory(
test_data_dir,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)#to set as training

valid_generator=train_datagen.flow_from_directory(
test_data_dir,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)#set to validation data

test_generator=train_datagen.flow_from_directory(
test_data_dir,
    target_size=(img_height,img_width),
    batch_size=1,
    class_mode='categorical',
    subset='training'
)
x,y=test_generator.next()
base_model=ResNet50(include_top=False,weights='imagenet')
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='sigmoid')(x)
predictions=Dense(train_generator.num_classes,activation='softmax')(x)
model=Model(inputs=base_model.input,outputs=predictions)#it becomes a transfer learn model


losses = []
accuracies = []
for layer in base_model.layers:
    layer.trainable=False
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
class LossAccHistory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            losses.append(logs['loss'])
            accuracies.append(logs['accuracy'])

# Create an instance of the callback
history_callback = LossAccHistory()
model.fit(train_generator,epochs=15,callbacks=[history_callback])


with open("Resnet-50.pkl", 'wb') as model_file:
    pickle.dump(model, model_file)
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from tensorflow.keras.applications.resnet50 import preprocess_input

# # Load and preprocess the image
# img_path = r"E:\Projects\Cancer_prediction\DL\input\normal\normal (122).png"# Replace with the path to your image
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)  # Preprocess the image for ResNet50

# # Make predictions
# predictions = model.predict(x)
# predictions
# #Interpret the predictions
# val=np.argmax(predictions)
# category(val)