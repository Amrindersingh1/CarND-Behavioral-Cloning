import csv
import cv2
import argparse
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout

class Pipeline:
    def __init__(self, base_path='', epochs=2):
        self.data = []
        self.epochs = epochs
        self.training_samples = []
        self.validation_samples = []
        self.correction_factor = 0.2
        self.base_path = base_path
        self.image_path = self.base_path + '/IMG/'
        self.driving_log_path = self.base_path + '/driving_log.csv'

    def import_data(self):
        with open(self.driving_log_path) as csvfile:
            reader = csv.reader(csvfile)
            # Skip the column names row
            next(reader)

            for line in reader:
                self.data.append(line)

        return None

    def process_batch(self, batch_sample):
        steering_angle = np.float32(batch_sample[3])
        images, steering_angles = [], []

        for image_path_index in range(3):
            image_name = batch_sample[image_path_index].split('/')[-1]

            image = cv2.imread(self.image_path + image_name)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cropped = rgb_image[60:130, :]
            resized = cv2.resize(cropped, (160,70))

            images.append(resized)

            if image_path_index == 1:
                steering_angles.append(steering_angle + self.correction_factor)
            elif image_path_index == 2:
                steering_angles.append(steering_angle - self.correction_factor)
            else:
                steering_angles.append(steering_angle)

            if image_path_index == 0:
                flipped_center_image = cv2.flip(resized, 1)
                images.append(flipped_center_image)
                steering_angles.append(-steering_angle)

        return images, steering_angles

    def data_generator(self, samples, batch_size=128):
        num_samples = len(samples)

        while True:
            shuffle(samples)

            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images, steering_angles = [], []

                for batch_sample in batch_samples:
                    augmented_images, augmented_angles = self.process_batch(batch_sample)
                    images.extend(augmented_images)
                    steering_angles.extend(augmented_angles)

                X_train, y_train = np.array(images), np.array(steering_angles)
                yield shuffle(X_train, y_train)

    def split_data(self):
        train, validation = train_test_split(self.data, test_size=0.2)
        self.training_samples, self.validation_samples = train, validation
        return None

    def train_generator(self, batch_size=128):
        return self.data_generator(samples=self.training_samples, batch_size=batch_size)

    def validation_generator(self, batch_size=128):
        return self.data_generator(samples=self.validation_samples, batch_size=batch_size)
    
    def model(optimizer='adam'):
        model = Sequential()
        model.add(Lambda(lambda x:  (x / 127.5) - 1., input_shape=(70, 160, 3)))
        model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
        model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
        model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))

        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam')

        return model

    def run(self):
        modell = self.model()
        self.split_data()
        modell.fit_generator(generator=self.train_generator(),
                                 validation_data=self.validation_generator(),
                                 epochs=self.epochs,
                                 steps_per_epoch=len(self.training_samples) * 2,
                                 validation_steps=len(self.validation_samples))
        modell.save('model.h5')

def main():
    # Instantiate the pipeline
    pipeline = Pipeline(base_path="./data/data", epochs=2)

    # Feed driving log data into the pipeline
    pipeline.import_data()
    # Start training
    pipeline.run()

if __name__ == '__main__':
    main()