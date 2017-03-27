import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import cv2
from random import shuffle

class Data_Generator:
    def __init__(self, metadata_path, batch_size, val_split):
        """
        Initialize image paths and gender labels
        # Arguments: metadata_path
        """
        self.metadata = scipy.io.loadmat(metadata_path)
        self.img_paths = self.metadata['wiki']['full_path'][0][0][0]
        self.genders = self.metadata['wiki']['gender'][0][0][0]
        self.batch_size = batch_size
        self.val_split = val_split
        self.load_keys()


    def load_keys(self):
        indices = []

        for i in range(len(self.img_paths)):
            if not np.isnan(self.genders[i]):
                indices.append(i)

        self.number_of_imgs = len(indices)

        self.val_size = int(self.number_of_imgs * self.val_split)
        self.train_size = self.number_of_imgs - self.val_size
        self.train_keys = indices[:self.train_size]
        self.val_keys = indices[self.train_size:]


    def load_data(self, is_train):
        """ loads wiki dataset 
        # Returns: faces and genders
                face: shape (64, 64, 1)
                gender_labels: 0 for female and 1 for male
        """
        while 1:
            faces = []
            gender_labels = []   

            shuffle(self.train_keys)
            keys = self.train_keys
            if not is_train:
                keys = self.val_keys 

            for key in keys:
                img_path = self.img_paths[key][0]
                img = cv2.imread('../wiki_crop/' + img_path, 0)            

                faces.append(cv2.resize(img, (64, 64)))
                gender_labels.append(self.genders[key])
                if len(faces) == self.batch_size:
                    faces = np.expand_dims(faces,-1)
                    gender_labels = pd.get_dummies(gender_labels).as_matrix()
                    yield (faces, gender_labels)
                    faces = []
                    gender_labels = []
            faces = np.expand_dims(faces,-1)
            gender_labels = pd.get_dummies(gender_labels).as_matrix()
            yield (faces, gender_labels)


