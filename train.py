"""
ML-MAP for ECoG channel prediction
Code written by: Harish RaviPrakash
This file is used for training models. Please see the README for details about training.
"""

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import math
from functools import partial
from os.path import join
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, LearningRateScheduler

K.set_image_data_format('channels_last')


def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1 + epoch) / float(epochs_drop)))


def get_callbacks(log_dir, model_name, check_dir, split):
    monitor_name = 'val_acc'
    csv_logger = CSVLogger(join(log_dir, model_name + 'split' + str(split) + '_log_' + '.csv'), separator=',')

    filepath = model_name + "_" + str(split) + "_{epoch:02d}.hdf5"
    model_checkpoint = ModelCheckpoint(join(check_dir, filepath),
                                       monitor=monitor_name, save_best_only=False, save_weights_only=False,
                                       verbose=1, mode='auto', period=10)
    reduce_lr = ReduceLROnPlateau(monitor=monitor_name, patience=20, mode='max', factor=1. / np.cbrt(2), min_lr=1e-5,
                                  verbose=2)
    return [model_checkpoint, reduce_lr, csv_logger]


def plot_training(training_history, savePath, model_name, fold):
    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))
    f.suptitle('TimeFreq2', fontsize=18)

    ax1.plot(training_history.history['acc'])
    ax1.plot(training_history.history['val_acc'])
    ax1.set_title('Channel Response')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(['Train', 'Val'], loc='upper left')
    ax1.legend(['Train'], loc='upper left')
    ax1.set_yticks(np.arange(0, 1.05, 0.05))
    ax1.set_xticks(np.arange(0, len(training_history.history['acc'])))
    ax1.grid(True)
    gridlines1 = ax1.get_xgridlines() + ax1.get_ygridlines()
    for line in gridlines1:
        line.set_linestyle('-.')

    ax2.plot(training_history.history['loss'])
    ax2.plot(training_history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend(['Train'], loc='upper right')
    ax1.set_xticks(np.arange(0, len(training_history.history['loss'])))
    ax2.grid(True)
    gridlines2 = ax2.get_xgridlines() + ax2.get_ygridlines()
    for line in gridlines2:
        line.set_linestyle('-.')

    f.savefig(join(savePath, model_name + '_fold_' + str(fold) + '_plots' + '.png'))
    plt.close()


def train(args, X_train, X2_train, X_val, X2_val, y_train, y_val, model, split):
    # Set the callbacks
    callbacks = get_callbacks(args.log_dir, args.model_name, args.check_dir, split)
    # Training the network
    history = model.fit([X_train, X2_train], y_train, validation_data=([X_val, X2_val], y_val), epochs=args.epochs,
                        batch_size=args.batch_size, callbacks=callbacks, verbose=1)

    # Plot the training data collected
    plot_training(history, args.save_dir, args.model_name, split)
    return model


def train2(args, X_train, X_val, y_train, y_val, model, split):
    # Set the callbacks
    callbacks = get_callbacks(args.log_dir, args.model_name, args.check_dir, split)
    # Training the network
    history = model.fit([X_train], y_train, validation_data=([X_val], y_val), epochs=args.epochs,
                        batch_size=args.batch_size, callbacks=callbacks, verbose=1)

    # Plot the training data collected
    plot_training(history, args.save_dir, args.model_name, split)
    return model
