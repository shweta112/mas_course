from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.optimizers import SGD
from models import model_CNN
from utils import Data_Generator
from math import ceil
import keras.backend as K
import tensorflow as tf

metadata_path = '../wiki/wiki.mat'
model_save_path = '../trained_models/model_CNN.hdf5'

num_classes = 2
image_size = (64, 64, 1)
batch_size = 150
num_epochs = 20
val_split = 0.1
data = Data_Generator(metadata_path, batch_size, val_split)
gen_train = data.load_data(True)
gen_val = data.load_data(False)

model = model_CNN(image_size, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
                                        metrics=['accuracy'])
csv_logger = CSVLogger('training.log')
early_stop = EarlyStopping('val_acc', patience=200, verbose=1)
model_checkpoint = ModelCheckpoint(model_save_path,
                                    'val_acc', verbose=0,
                                    save_best_only=True)

model_callbacks = [early_stop, model_checkpoint, csv_logger]

#keras bug 
K.get_session().run(tf.global_variables_initializer())
model.fit_generator(gen_train, nb_epoch=num_epochs, verbose=1, 
                                    validation_data=gen_val,
                                    samples_per_epoch=data.train_size,
                                    nb_val_samples=data.val_size,
                                    callbacks=model_callbacks)
