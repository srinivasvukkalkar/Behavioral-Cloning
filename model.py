import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from keras.regularizers import l2

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)
        
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.15)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = './data/IMG/' + filename
                    image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)
                    images.append(image)
                    images.append(cv2.flip(image,1))
                    if(i==0):
                        measurement = float(batch_sample[3])
                    elif(i==1):
                        measurement = float(batch_sample[3])+0.25
                    elif(i==2):
                        measurement = float(batch_sample[3])-0.25
                    measurements.append(measurement)
                    measurements.append(measurement*-1.0)

#             augmented_images, augmented_measurements = [], []
#             for image,measurement in zip(images, measurements):
#                 augmented_images.append(image)
#                 augmented_measurements.append(measurement)
#                 augmented_images.append(cv2.flip(image,1))
#                 augmented_measurements.append(measurement*-1.0)

            X_train = np.array(images)
            y_train = np.array(measurements)
            
            yield sklearn.utils.shuffle(X_train, y_train)

#Set batch size
batch_size=32

#compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# ch, row, col = 3, 80, 320  #Trimmed image format


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

print("Starting Architecture")

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,5,5,subsample=(2,2),activation="elu"))
model.add(Conv2D(36,5,5,subsample=(2,2),activation="elu"))
model.add(Conv2D(48,5,5,subsample=(2,2),activation="elu"))
model.add(Conv2D(64,3,3,activation="elu"))
# model.add(MaxPooling2D())
model.add(Conv2D(64,3,3,activation="elu"))
# model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100,activation="elu"))
model.add(Dropout(0.25))
model.add(Dense(50,activation="elu"))
# model.add(Dropout(0.2))
model.add(Dense(10,activation="elu"))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
# model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/32, epochs=5, validation_data=validation_generator,   /                                validation_steps=len(validation_samples)/32, verbose=1)

model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size, validation_data=validation_generator, 
            validation_steps=len(validation_samples)/batch_size, epochs=5, verbose=1)

# keras method to print the model summary
model.summary()

model.save('model.h5')

print("Model ran successfully")
exit()