import os
from os import environ
import numpy as np
import argparse
from loadData import read_data, blockToChannel, reshapeData, reshapeOutputs, reshapeAR, predictions, \
    predictions2, probBlockApproach, earlyFusion, reshapeTimeFeats
from sklearn.model_selection import StratifiedShuffleSplit
from modelFiles2 import create_model
from train import train2
from keras.utils import to_categorical


# To compute statistics
def getstats(out, gt):
    TP = np.sum(out[gt == 1])
    FP = np.sum(out[gt == 0])
    FN = np.sum(gt[out == 0])
    TN = np.sum(1-(out[gt == 0]))
    sensivity = float(TP)/(TP+FN+0.00001)
    specificity = float(TN)/(TN+FP+0.00001)
    precision = float(TP)/(TP+FP+0.00001)
    accuracy = float(TP+TN)/(TP+FP+TN+FN+0.00001)
    return [sensivity, specificity, precision, accuracy]


def main(args):
    try:
        os.makedirs(args.log_dir)
    except:
        pass
    try:
        os.makedirs(args.check_dir)
    except:
        pass

    # Load the data
    X_all, X2_all, Y_all, Y, blockSize = read_data(args.timeFeats, args.data_root_dir, args.file1, args.file2)
    if os.path.isfile(os.path.join(args.save_dir, 'dataSplits_ratio2.npz')):
        with np.load(os.path.join(args.save_dir, 'dataSplits_ratio2.npz'), allow_pickle=True) as data:
            train_ind = data['train_ind']
            val_ind = data['val_ind']
    else:
        train_ind = []
        val_ind = []
        # defining hold-out ratio
        ss = StratifiedShuffleSplit(n_splits=10, test_size=0.1)
        for train_indx, val_indx in ss.split(np.zeros((Y.shape[0], 1)), Y):
            train_ind.append(train_indx)
            val_ind.append(val_indx)
        np.savez_compressed(os.path.join(args.save_dir, 'dataSplits_ratio2.npz'),
                            train_ind=train_ind, val_ind=val_ind)
    # Start with the first fold
    for i in range(10):
        train_indx = train_ind[i]
        val_indx = val_ind[i]
        # Splitting into training & validation
        X_train, y_train = reshapeTimeFeats(X_all, Y_all, train_indx, blockSize)
        X3_train, X_mean, X_std = earlyFusion(X_train)
        y_train = to_categorical(y_train, 2)
        X_val, y_val = reshapeTimeFeats(X_all, Y_all, val_indx, blockSize)
        X3_val, _, _ = earlyFusion(X_val, X_mean, X_std, training=False)
        y_val = to_categorical(y_val, 2)

        # Create the model
        model = create_model(X3_train)
        print(model.summary())

        # Train the model
        model = train2(args, X3_train, X3_val, y_train, y_val, model, i)
        # Save the model
        model.save(os.path.join(args.save_dir, args.model_name + 'fold_' + str(i) + '.h5'))
        # Initializing output variables
        output = []

        # Evaluating the model
        scores = model.evaluate([X3_val], y_val, verbose=2)
        print('Model PSDTime')
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        print(scores)
        for j in range(X3_val.shape[0]):
            tmp1 = X3_val[j, ...]
            tmp1 = np.reshape(tmp1, (1, X3_val.shape[1], X3_val.shape[2]))
            output.append(model.predict(tmp1))
        output = np.array(output)
        output = np.reshape(output, (output.shape[0], output.shape[2]))
        np.savez_compressed(
            os.path.join(args.save_dir, args.model_name + 'fold_' + str(i) + 'output' + '.npz'), block_Scores=scores,
            block_op=output, block_gt=y_val)
        output_reshaped = reshapeOutputs(output)
        output_reshaped2 = probBlockApproach(output, blockSize)
        pred = blockToChannel(output_reshaped, blockSize)
        # Compute different weighting i.e. not majority voting but weighted classification
        pred2 = predictions(output_reshaped2)
        pred3 = predictions2(output_reshaped2)
        [Se, Sp, Pr, acc] = getstats(pred, Y[val_indx])
        [Se2, Sp2, Pr2, acc2] = getstats(pred2, Y[val_indx])
        [Se3, Sp3, Pr3, acc3] = getstats(pred3, Y[val_indx])
        np.savez_compressed(
            os.path.join(args.save_dir, args.model_name + 'fold_' + str(i) + 'Finaloutput' + '.npz'), block_Scores=scores,
            block_op=output_reshaped, block_gt=y_val, pred=pred, gt=Y[val_indx], Se=Se, Sp=Sp, Pr=Pr, acc=acc, pred2=pred2,
            Se2=Se2, Sp2=Sp2, Pr2=Pr2, acc2=acc2, pred3=pred3, Se3=Se3, Sp3=Sp3, Pr3=Pr3, acc3=acc3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train on Medical Data')
    parser.add_argument('--data_root_dir', type=str, default='',
                        help='Path to time series data')
    parser.add_argument('--file1', type=str, default='/Path/to/PSD.mat',
                        help='Path to time series data')
    parser.add_argument('--file2', type=str, default='/Path/to/blockLabels.txt',
                        help='Path to sub-block labels')
    parser.add_argument('--timeFeats', type=list, default='',
                        choices=['Activity', 'Mobility', 'P2P', 'Complexity', 'Mean', 'Skew', 'Kurtosis']
                        help='Time feature file names')
    parser.add_argument('--model_name', type=str, default='',
                        help='Model name to store')
    parser.add_argument('--save_dir', type=str,
                        default='',
                        help='Path to store trained model')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training/testing.')
    parser.add_argument('--epochs', type=int, default=900,
                        help='Number of epochs for training of the model.')
    parser.add_argument('--shuffle_data', type=bool, default=True,
                        help='Shuffle batches during training and validation.')
    parser.add_argument('--log_dir', type=str, default='',
                        help='Path to store training logs')
    parser.add_argument('--check_dir', type=str,
                        default='',
                        help='Path to store trained models')
    arguments = parser.parse_args()
    environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Selecting GPU on Newton cluster
    environ["CUDA_VISIBLE_DEVICES"] = str(0)
    main(arguments)
