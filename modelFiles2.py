from keras import layers
from keras.models import Model
from keras import regularizers
from customLayers2 import AttentionLSTM
from keras.optimizers import Adam

def create_model(X1):
    """
    Only time domain as input
    :param X1: Time features
    :return: model
    """
    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))
    # LSTM path of the network
    x = layers.LSTM(8)(sig_input)
    dp = layers.Dropout(0.8)(x)

    # Fully convolutional path of the network
    x1 = layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform',
                       activity_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.2)(act1)

    x2 = layers.Conv1D(64, 5, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.2)(act2)

    x3 = layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.2)(act3)

    y = layers.GlobalAveragePooling1D()(dp3)

    out = layers.Dense(2, activation='softmax')(y)
    model = Model(inputs=[sig_input], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model2(X1, X2):
    """
    No layers in the fusion path
    :param X1: Time features
    :param X2: PSD features
    :return: model
    """
    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))
    sig_input2 = layers.Input(shape=(X2.shape[1],))
    # LSTM path of the network
    x = layers.LSTM(8)(sig_input)
    dp = layers.Dropout(0.8)(x)

    # Fully convolutional path of the network
    x1 = layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform', activity_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.2)(act1)

    x2 = layers.Conv1D(64, 5, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.2)(act2)

    x3 = layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.2)(act3)

    y = layers.GlobalAveragePooling1D()(dp3)

    # AR features
    y1 = layers.Dense(64, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(sig_input2)
    bn4 = layers.BatchNormalization()(y1)
    act4 = layers.LeakyReLU(alpha=0.1)(bn4)
    dp4 = layers.Dropout(0.2)(act4)

    # Concatenation of LSTM and FCN paths
    z = layers.concatenate([dp, y, dp4])

    out = layers.Dense(2, activation='softmax')(z)
    model = Model(inputs=[sig_input, sig_input2], outputs=out)
    opt = Adam(lr=.0005)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model3(X1, X2):
    """
    Linear activation 1D conv in the fusion path
    :param X1: Time features
    :param X2: PSD features
    :return: model
    """
    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))
    sig_input2 = layers.Input(shape=(X2.shape[1],))
    # LSTM path of the network
    x = layers.LSTM(8)(sig_input)
    dp = layers.Dropout(0.8)(x)

    # Fully convolutional path of the network
    x1 = layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform', activity_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.2)(act1)

    x2 = layers.Conv1D(64, 5, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.2)(act2)

    x3 = layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.2)(act3)

    y = layers.GlobalAveragePooling1D()(dp3)

    # AR features
    y1 = layers.Dense(64, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(sig_input2)
    bn4 = layers.BatchNormalization()(y1)
    act4 = layers.LeakyReLU(alpha=0.1)(bn4)
    dp4 = layers.Dropout(0.2)(act4)

    # Concatenation of LSTM and FCN paths
    z = layers.concatenate([dp, y, dp4])

    # Reshape to do 1D conv for feature sparsification
    z1 = layers.Reshape((200, 1))(z)
    z2 = layers.Conv1D(64, 5, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(z1)
    z3 = layers.Conv1D(32, 3, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(z2)
    z4 = layers.GlobalAveragePooling1D()(z3)
    out = layers.Dense(2, activation='softmax')(z4)
    model = Model(inputs=[sig_input, sig_input2], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model4(X1, X2):
    """
    ReLU activation and BN+DP 1D conv in the fusion path
    :param X1: Time features
    :param X2: PSD features
    :return: model
    """

    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))
    sig_input2 = layers.Input(shape=(X2.shape[1],))
    # LSTM path of the network
    x = layers.LSTM(8)(sig_input)
    dp = layers.Dropout(0.8)(x)

    # Fully convolutional path of the network
    x1 = layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform', activity_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.2)(act1)

    x2 = layers.Conv1D(64, 5, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.2)(act2)

    x3 = layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.2)(act3)

    y = layers.GlobalAveragePooling1D()(dp3)

    # AR features
    y1 = layers.Dense(64, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(sig_input2)
    bn4 = layers.BatchNormalization()(y1)
    act4 = layers.LeakyReLU(alpha=0.1)(bn4)
    dp4 = layers.Dropout(0.2)(act4)

    # Concatenation of LSTM and FCN paths
    z = layers.concatenate([dp, y, dp4])

    # Reshape to do 1D conv for feature sparsification
    z1 = layers.Reshape((200, 1))(z)
    z2 = layers.Conv1D(64, 5, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(z1)
    bn5 = layers.BatchNormalization()(z2)
    act5 = layers.LeakyReLU(alpha=0.1)(bn5)
    dp5 = layers.Dropout(0.2)(act5)
    z3 = layers.Conv1D(32, 3, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp5)
    bn6 = layers.BatchNormalization()(z3)
    act6 = layers.LeakyReLU(alpha=0.1)(bn6)
    dp6 = layers.Dropout(0.2)(act6)
    z4 = layers.GlobalAveragePooling1D()(dp6)
    out = layers.Dense(2, activation='softmax')(z4)

    model = Model(inputs=[sig_input, sig_input2], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model5(X1, X2):
    """
    Attention LSTM instead of LSTM
    :param X1: Time features
    :param X2: PSD features
    :return: model
    """
    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))
    sig_input2 = layers.Input(shape=(X2.shape[1],))
    # LSTM path of the network
    x = AttentionLSTM(8)(sig_input)
    dp = layers.Dropout(0.8)(x)

    # Fully convolutional path of the network
    x1 = layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform', activity_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.2)(act1)

    x2 = layers.Conv1D(64, 5, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.2)(act2)

    x3 = layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.2)(act3)

    y = layers.GlobalAveragePooling1D()(dp3)

    # AR features
    y1 = layers.Dense(64, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(sig_input2)
    bn4 = layers.BatchNormalization()(y1)
    act4 = layers.LeakyReLU(alpha=0.1)(bn4)
    dp4 = layers.Dropout(0.2)(act4)

    # Concatenation of LSTM and FCN paths
    z = layers.concatenate([dp, y, dp4])

    # Reshape to do 1D conv for feature sparsification
    z1 = layers.Reshape((200, 1))(z)
    z2 = layers.Conv1D(64, 5, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(z1)
    bn5 = layers.BatchNormalization()(z2)
    act5 = layers.LeakyReLU(alpha=0.1)(bn5)
    dp5 = layers.Dropout(0.2)(act5)
    z3 = layers.Conv1D(32, 3, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp5)
    bn6 = layers.BatchNormalization()(z3)
    act6 = layers.LeakyReLU(alpha=0.1)(bn6)
    dp6 = layers.Dropout(0.2)(act6)
    z4 = layers.GlobalAveragePooling1D()(dp6)
    out = layers.Dense(2, activation='softmax')(z4)

    model = Model(inputs=[sig_input, sig_input2], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model6(X1, X2):
    """
    No activity regularizer in the first 1D conv layer in LSTM-FCN
    :param X1: Time features
    :param X2: PSD features
    :return: model
    """
    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))
    sig_input2 = layers.Input(shape=(X2.shape[1],))
    # LSTM path of the network
    x = layers.LSTM(8)(sig_input)
    dp = layers.Dropout(0.8)(x)

    # Fully convolutional path of the network
    x1 = layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.2)(act1)

    x2 = layers.Conv1D(64, 5, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.2)(act2)

    x3 = layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.2)(act3)

    y = layers.GlobalAveragePooling1D()(dp3)

    # AR features
    y1 = layers.Dense(64, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(sig_input2)
    bn4 = layers.BatchNormalization()(y1)
    act4 = layers.LeakyReLU(alpha=0.1)(bn4)
    dp4 = layers.Dropout(0.2)(act4)

    # Concatenation of LSTM and FCN paths
    z = layers.concatenate([dp, y, dp4])

    # Reshape to do 1D conv for feature sparsification
    z1 = layers.Reshape((200, 1))(z)
    z2 = layers.Conv1D(64, 5, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(z1)
    bn5 = layers.BatchNormalization()(z2)
    act5 = layers.LeakyReLU(alpha=0.1)(bn5)
    dp5 = layers.Dropout(0.2)(act5)
    z3 = layers.Conv1D(32, 3, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp5)
    bn6 = layers.BatchNormalization()(z3)
    act6 = layers.LeakyReLU(alpha=0.1)(bn6)
    dp6 = layers.Dropout(0.2)(act6)
    z4 = layers.GlobalAveragePooling1D()(dp6)
    out = layers.Dense(2, activation='softmax')(z4)

    model = Model(inputs=[sig_input, sig_input2], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model7(X1, X2):
    """
    3 Conv1D layers in the fusion network
    :param X1: Time domain features
    :param X2: PSD features
    :return: model
    """
    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))
    sig_input2 = layers.Input(shape=(X2.shape[1],))
    # LSTM path of the network
    x = layers.LSTM(8)(sig_input)
    dp = layers.Dropout(0.8)(x)

    # Fully convolutional path of the network
    x1 = layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform',
                       activity_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.2)(act1)

    x2 = layers.Conv1D(64, 5, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.2)(act2)

    x3 = layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.2)(act3)

    y = layers.GlobalAveragePooling1D()(dp3)

    # PSD features
    y1 = layers.Dense(64, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(sig_input2)
    bn4 = layers.BatchNormalization()(y1)
    act4 = layers.LeakyReLU(alpha=0.1)(bn4)
    dp4 = layers.Dropout(0.2)(act4)

    # Concatenation of LSTM and FCN paths
    z = layers.concatenate([dp, y, dp4])

    # Reshape to do 1D conv for feature sparsification
    z1 = layers.Reshape((200, 1))(z)
    z2 = layers.Conv1D(64, 5, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(z1)
    bn5 = layers.BatchNormalization()(z2)
    act5 = layers.LeakyReLU(alpha=0.1)(bn5)
    dp5 = layers.Dropout(0.2)(act5)
    z3 = layers.Conv1D(32, 3, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp5)
    bn6 = layers.BatchNormalization()(z3)
    act6 = layers.LeakyReLU(alpha=0.1)(bn6)
    dp6 = layers.Dropout(0.2)(act6)
    z4 = layers.Conv1D(32, 3, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp6)
    bn7 = layers.BatchNormalization()(z4)
    act7 = layers.LeakyReLU(alpha=0.1)(bn7)
    dp7 = layers.Dropout(0.2)(act7)
    z5 = layers.GlobalAveragePooling1D()(dp7)
    out = layers.Dense(2, activation='softmax')(z5)

    model = Model(inputs=[sig_input, sig_input2], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model8(X1, X2):
    """
    Dense layer after 2 Conv1D layers in the fusion network
    :param X1: Time domain features
    :param X2: PSD features
    :return: model
    """
    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))
    sig_input2 = layers.Input(shape=(X2.shape[1],))
    # LSTM path of the network
    x = layers.LSTM(8)(sig_input)
    dp = layers.Dropout(0.8)(x)

    # Fully convolutional path of the network
    x1 = layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform',
                       activity_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.2)(act1)

    x2 = layers.Conv1D(64, 5, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.2)(act2)

    x3 = layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.2)(act3)

    y = layers.GlobalAveragePooling1D()(dp3)

    # PSD features
    y1 = layers.Dense(64, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(sig_input2)
    bn4 = layers.BatchNormalization()(y1)
    act4 = layers.LeakyReLU(alpha=0.1)(bn4)
    dp4 = layers.Dropout(0.2)(act4)

    # Concatenation of LSTM and FCN paths
    z = layers.concatenate([dp, y, dp4])

    # Reshape to do 1D conv for feature sparsification
    z1 = layers.Reshape((200, 1))(z)
    z2 = layers.Conv1D(64, 5, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(z1)
    bn5 = layers.BatchNormalization()(z2)
    act5 = layers.LeakyReLU(alpha=0.1)(bn5)
    dp5 = layers.Dropout(0.2)(act5)
    z3 = layers.Conv1D(32, 3, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp5)
    bn6 = layers.BatchNormalization()(z3)
    act6 = layers.LeakyReLU(alpha=0.1)(bn6)
    dp6 = layers.Dropout(0.2)(act6)
    z5 = layers.GlobalAveragePooling1D()(dp6)
    z6 = layers.Dense(32, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(z5)
    bn7 = layers.BatchNormalization()(z6)
    act7 = layers.LeakyReLU(alpha=0.1)(bn7)
    dp7 = layers.Dropout(0.2)(act7)
    out = layers.Dense(2, activation='softmax')(dp7)

    model = Model(inputs=[sig_input, sig_input2], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model9(X1, X2):
    """
    Dense layer before the 2 Conv1D layers in the fusion network
    :param X1: Time domain features
    :param X2: PSD features
    :return: model
    """
    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))
    sig_input2 = layers.Input(shape=(X2.shape[1],))
    # LSTM path of the network
    x = layers.LSTM(8)(sig_input)
    dp = layers.Dropout(0.8)(x)

    # Fully convolutional path of the network
    x1 = layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform',
                       activity_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.2)(act1)

    x2 = layers.Conv1D(64, 5, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.2)(act2)

    x3 = layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.2)(act3)

    y = layers.GlobalAveragePooling1D()(dp3)

    # PSD features
    y1 = layers.Dense(64, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(sig_input2)
    bn4 = layers.BatchNormalization()(y1)
    act4 = layers.LeakyReLU(alpha=0.1)(bn4)
    dp4 = layers.Dropout(0.2)(act4)

    # Concatenation of LSTM and FCN paths
    z = layers.concatenate([dp, y, dp4])

    # Reshape to do 1D conv for feature sparsification
    z1 = layers.Dense(64, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(z)
    bn5 = layers.BatchNormalization()(z1)
    act5 = layers.LeakyReLU(alpha=0.1)(bn5)
    dp5 = layers.Dropout(0.2)(act5)
    z2 = layers.Reshape((64, 1))(dp5)
    z3 = layers.Conv1D(64, 5, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(z2)
    bn6 = layers.BatchNormalization()(z3)
    act6 = layers.LeakyReLU(alpha=0.1)(bn6)
    dp6 = layers.Dropout(0.2)(act6)
    z4 = layers.Conv1D(32, 3, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp6)
    bn7 = layers.BatchNormalization()(z4)
    act7 = layers.LeakyReLU(alpha=0.1)(bn7)
    dp7 = layers.Dropout(0.2)(act7)
    z5 = layers.GlobalAveragePooling1D()(dp7)
    out = layers.Dense(2, activation='softmax')(z5)

    model = Model(inputs=[sig_input, sig_input2], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model10(X1, X2):
    """
    Dilated convolutions in fusion network
    :param X1: Time features
    :param X2: PSD features
    :return: model
    """

    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))
    sig_input2 = layers.Input(shape=(X2.shape[1],))
    # LSTM path of the network
    x = layers.LSTM(8)(sig_input)
    dp = layers.Dropout(0.8)(x)

    # Fully convolutional path of the network
    x1 = layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform', activity_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.2)(act1)

    x2 = layers.Conv1D(64, 5, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.2)(act2)

    x3 = layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.2)(act3)

    y = layers.GlobalAveragePooling1D()(dp3)

    # AR features
    y1 = layers.Dense(64, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(sig_input2)
    bn4 = layers.BatchNormalization()(y1)
    act4 = layers.LeakyReLU(alpha=0.1)(bn4)
    dp4 = layers.Dropout(0.2)(act4)

    # Concatenation of LSTM and FCN paths
    z = layers.concatenate([dp, y, dp4])

    # Reshape to do 1D conv for feature sparsification
    z1 = layers.Reshape((200, 1))(z)
    z2 = layers.Conv1D(64, 5, dilation_rate=4, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(z1)
    bn5 = layers.BatchNormalization()(z2)
    act5 = layers.LeakyReLU(alpha=0.1)(bn5)
    dp5 = layers.Dropout(0.2)(act5)
    z3 = layers.Conv1D(32, 3, dilation_rate=2, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp5)
    bn6 = layers.BatchNormalization()(z3)
    act6 = layers.LeakyReLU(alpha=0.1)(bn6)
    dp6 = layers.Dropout(0.2)(act6)
    z4 = layers.GlobalAveragePooling1D()(dp6)
    out = layers.Dense(2, activation='softmax')(z4)

    model = Model(inputs=[sig_input, sig_input2], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model11(X1, X2):
    """
    Replacing 1D conv by Dense layers in fusion network
    :param X1: Time domain features
    :param X2: PSD features
    :return: model
    """
    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))
    sig_input2 = layers.Input(shape=(X2.shape[1],))
    # LSTM path of the network
    x = layers.LSTM(8)(sig_input)
    dp = layers.Dropout(0.8)(x)

    # Fully convolutional path of the network
    x1 = layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform',
                       activity_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.2)(act1)

    x2 = layers.Conv1D(64, 5, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.2)(act2)

    x3 = layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.2)(act3)

    y = layers.GlobalAveragePooling1D()(dp3)

    # PSD features
    y1 = layers.Dense(64, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(sig_input2)
    bn4 = layers.BatchNormalization()(y1)
    act4 = layers.LeakyReLU(alpha=0.1)(bn4)
    dp4 = layers.Dropout(0.2)(act4)

    # Concatenation of LSTM and FCN paths
    z = layers.concatenate([dp, y, dp4])

    # Fully connected layer to learn associations between time and frequency data
    z1 = layers.Dense(64, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(z)
    bn5 = layers.BatchNormalization()(z1)
    act5 = layers.LeakyReLU(alpha=0.1)(bn5)
    dp5 = layers.Dropout(0.2)(act5)
    z2 = layers.Dense(32, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp5)
    bn6 = layers.BatchNormalization()(z2)
    act6 = layers.LeakyReLU(alpha=0.1)(bn6)
    dp6 = layers.Dropout(0.2)(act6)
    out = layers.Dense(2, activation='softmax')(dp6)

    model = Model(inputs=[sig_input, sig_input2], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model12(X1, X2):
    """
    Increasing dense layers to 2 in PSD feature extraction
    :param X1: Time domain features
    :param X2: PSD features
    :return: model
    """

    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))
    sig_input2 = layers.Input(shape=(X2.shape[1],))
    # LSTM path of the network
    x = layers.LSTM(8)(sig_input)
    dp = layers.Dropout(0.8)(x)

    # Fully convolutional path of the network
    x1 = layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform',
                       activity_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.2)(act1)

    x2 = layers.Conv1D(64, 5, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.2)(act2)

    x3 = layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.2)(act3)

    y = layers.GlobalAveragePooling1D()(dp3)

    # PSD features
    y1 = layers.Dense(128, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(sig_input2)
    bn4 = layers.BatchNormalization()(y1)
    act4 = layers.LeakyReLU(alpha=0.1)(bn4)
    dp4 = layers.Dropout(0.2)(act4)
    y2 = layers.Dense(64, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp4)
    bn9 = layers.BatchNormalization()(y2)
    act9 = layers.LeakyReLU(alpha=0.1)(bn9)
    dp9 = layers.Dropout(0.2)(act9)

    # Concatenation of LSTM and FCN paths
    z = layers.concatenate([dp, y, dp9])

    # Reshape to do 1D conv for feature sparsification
    z1 = layers.Reshape((200, 1))(z)
    z2 = layers.Conv1D(64, 5, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(z1)
    bn5 = layers.BatchNormalization()(z2)
    act5 = layers.LeakyReLU(alpha=0.1)(bn5)
    dp5 = layers.Dropout(0.2)(act5)
    z3 = layers.Conv1D(32, 3, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp5)
    bn6 = layers.BatchNormalization()(z3)
    act6 = layers.LeakyReLU(alpha=0.1)(bn6)
    dp6 = layers.Dropout(0.2)(act6)
    z4 = layers.GlobalAveragePooling1D()(dp6)
    out = layers.Dense(2, activation='softmax')(z4)

    model = Model(inputs=[sig_input, sig_input2], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model13(X1):
    """
    Only time domain as input
    :param X1: Time features
    :return: model
    """
    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))

    # Fully convolutional path of the network
    x1 = layers.Conv1D(64, 10, padding='same', kernel_initializer='he_uniform',
                       activity_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.2)(act1)

    x2 = layers.Conv1D(256, 10, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.2)(act2)

    x3 = layers.Conv1D(128, 10, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.2)(act3)

    y = layers.GlobalAveragePooling1D()(dp3)

    out = layers.Dense(2, activation='softmax')(y)
    model = Model(inputs=[sig_input], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model14(X1):
    """
    Only time domain as input
    :param X1: Time features
    :return: model
    """
    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))

    # Fully convolutional path of the network
    x1 = layers.Conv1D(64, 10, padding='same', kernel_initializer='he_uniform',
                       activity_regularizer=regularizers.l2(0.001), activation='relu')(sig_input)
    mp1 = layers.MaxPool1D(pool_size=4, strides=4)(x1)

    x2 = layers.Conv1D(256, 10, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001), activation='relu')(mp1)
    mp2 = layers.MaxPool1D(pool_size=4, strides=4)(x2)

    x3 = layers.Conv1D(128, 10, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001), activation='relu')(mp2)
    mp3 = layers.MaxPool1D(pool_size=4, strides=4)(x3)

    y = layers.Flatten()(mp3)

    y1 = layers.Dense(units=256, activation='relu')(y)
    dp1 = layers.Dropout(0.5)(y1)
    out = layers.Dense(2, activation='softmax')(dp1)
    model = Model(inputs=[sig_input], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model15(X1):
    """
    Only time domain as input
    :param X1: Time features
    :return: model
    """
    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))

    # Fully convolutional path of the network
    x1 = layers.Conv1D(64, 10, padding='same', kernel_initializer='he_uniform',
                       activity_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.5)(act1)
    mp1 = layers.MaxPool1D(pool_size=4, strides=4)(dp1)

    x2 = layers.Conv1D(256, 10, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(mp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.5)(act2)
    mp2 = layers.MaxPool1D(pool_size=4, strides=4)(dp2)

    x3 = layers.Conv1D(128, 10, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(mp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.5)(act3)
    mp3 = layers.MaxPool1D(pool_size=4, strides=4)(dp3)

    y = layers.Flatten()(mp3)

    y1 = layers.Dense(units=256, activation='relu')(y)
    dp1 = layers.Dropout(0.5)(y1)
    out = layers.Dense(2, activation='softmax')(dp1)
    model = Model(inputs=[sig_input], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model16(X1):
    """
    Only time domain as input
    :param X1: Time features
    :return: model
    """
    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))

    # Fully convolutional path of the network
    x1 = layers.Conv1D(64, 10, padding='same', kernel_initializer='he_uniform',
                       activation='relu')(sig_input)
    mp1 = layers.MaxPool1D(pool_size=4, strides=4)(x1)

    x2 = layers.Conv1D(256, 10, padding='same', kernel_initializer='he_uniform',
                       activation='relu')(mp1)
    mp2 = layers.MaxPool1D(pool_size=4, strides=4)(x2)

    x3 = layers.Conv1D(128, 10, padding='same', kernel_initializer='he_uniform',
                       activation='relu')(mp2)
    mp3 = layers.MaxPool1D(pool_size=4, strides=4)(x3)

    y = layers.Flatten()(mp3)

    y1 = layers.Dense(units=256, activation='relu')(y)
    dp1 = layers.Dropout(0.5)(y1)
    out = layers.Dense(2, activation='softmax')(dp1)
    model = Model(inputs=[sig_input], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model17(X1):
    """
    Only time domain as input
    :param X1: Time features
    :return: model
    """
    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))

    # Fully convolutional path of the network
    x1 = layers.Conv1D(64, 10, padding='same', kernel_initializer='he_uniform',
                       activity_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.5)(act1)
    mp1 = layers.MaxPool1D(pool_size=4, strides=4)(dp1)

    x2 = layers.Conv1D(256, 10, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(mp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.5)(act2)
    mp2 = layers.MaxPool1D(pool_size=4, strides=4)(dp2)

    x3 = layers.Conv1D(128, 10, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(mp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.5)(act3)
    mp3 = layers.MaxPool1D(pool_size=4, strides=4)(dp3)

    y = layers.GlobalAveragePooling1D()(mp3)

    y1 = layers.Dense(units=256, activation='relu')(y)
    dp1 = layers.Dropout(0.5)(y1)
    out = layers.Dense(2, activation='softmax')(dp1)
    model = Model(inputs=[sig_input], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model18(X1, X2):
    """
    ReLU activation and BN+DP 1D conv in the fusion path
    :param X1: Time features
    :param X2: PSD features
    :return: model
    """

    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))
    sig_input2 = layers.Input(shape=(X2.shape[1],))
    # LSTM path of the network
    # Fully convolutional path of the network
    x1 = layers.Conv1D(64, 10, padding='same', kernel_initializer='he_uniform',
                       activity_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.5)(act1)
    mp1 = layers.MaxPool1D(pool_size=4, strides=4)(dp1)

    x2 = layers.Conv1D(256, 10, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(mp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.5)(act2)
    mp2 = layers.MaxPool1D(pool_size=4, strides=4)(dp2)

    x3 = layers.Conv1D(128, 10, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(mp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.5)(act3)
    mp3 = layers.MaxPool1D(pool_size=4, strides=4)(dp3)

    y = layers.GlobalAveragePooling1D()(mp3)

    # PSD features
    y1 = layers.Dense(64, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(sig_input2)
    bn4 = layers.BatchNormalization()(y1)
    act4 = layers.LeakyReLU(alpha=0.1)(bn4)
    dp4 = layers.Dropout(0.2)(act4)

    # Concatenation of LSTM and FCN paths
    z = layers.concatenate([y, dp4])

    # Reshape to do 1D conv for feature sparsification
    z1 = layers.Reshape((192, 1))(z)
    z2 = layers.Conv1D(64, 5, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(z1)
    bn5 = layers.BatchNormalization()(z2)
    act5 = layers.LeakyReLU(alpha=0.1)(bn5)
    dp5 = layers.Dropout(0.2)(act5)
    z3 = layers.Conv1D(32, 3, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp5)
    bn6 = layers.BatchNormalization()(z3)
    act6 = layers.LeakyReLU(alpha=0.1)(bn6)
    dp6 = layers.Dropout(0.2)(act6)
    z4 = layers.GlobalAveragePooling1D()(dp6)
    out = layers.Dense(2, activation='softmax')(z4)

    model = Model(inputs=[sig_input, sig_input2], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model19(X1, X2):
    """
    ReLU activation and BN+DP 1D conv in the fusion path
    :param X1: Time features
    :param X2: PSD features
    :return: model
    """

    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))
    sig_input2 = layers.Input(shape=(X2.shape[1],))
    # LSTM path of the network
    # Fully convolutional path of the network
    x1 = layers.Conv1D(64, 10, padding='same', kernel_initializer='he_uniform',
                       activity_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.2)(act1)

    x2 = layers.Conv1D(256, 10, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.2)(act2)

    x3 = layers.Conv1D(128, 10, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.2)(act3)

    y = layers.GlobalAveragePooling1D()(dp3)

    # PSD features
    y1 = layers.Dense(64, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(sig_input2)
    bn4 = layers.BatchNormalization()(y1)
    act4 = layers.LeakyReLU(alpha=0.1)(bn4)
    dp4 = layers.Dropout(0.2)(act4)

    # Concatenation of LSTM and FCN paths
    z = layers.concatenate([y, dp4])

    # Reshape to do 1D conv for feature sparsification
    z1 = layers.Reshape((192, 1))(z)
    z2 = layers.Conv1D(64, 5, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(z1)
    bn5 = layers.BatchNormalization()(z2)
    act5 = layers.LeakyReLU(alpha=0.1)(bn5)
    dp5 = layers.Dropout(0.2)(act5)
    z3 = layers.Conv1D(32, 3, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp5)
    bn6 = layers.BatchNormalization()(z3)
    act6 = layers.LeakyReLU(alpha=0.1)(bn6)
    dp6 = layers.Dropout(0.2)(act6)
    z4 = layers.GlobalAveragePooling1D()(dp6)
    out = layers.Dense(2, activation='softmax')(z4)

    model = Model(inputs=[sig_input, sig_input2], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model20(X1, X2):
    """
    ReLU activation and BN+DP 1D conv in the fusion path
    :param X1: Time features
    :param X2: PSD features
    :return: model
    """

    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))
    sig_input2 = layers.Input(shape=(X2.shape[1],))
    # LSTM path of the network
    # Fully convolutional path of the network
    x1 = layers.Conv1D(64, 10, padding='same', kernel_initializer='he_uniform',
                       activity_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.2)(act1)

    x2 = layers.Conv1D(256, 10, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.2)(act2)

    x3 = layers.Conv1D(128, 10, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.2)(act3)

    y = layers.GlobalAveragePooling1D()(dp3)

    # PSD features
    y1 = layers.Dense(128, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(sig_input2)
    bn4 = layers.BatchNormalization()(y1)
    act4 = layers.LeakyReLU(alpha=0.1)(bn4)
    dp4 = layers.Dropout(0.2)(act4)
    y2 = layers.Dense(64, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp4)
    bn4 = layers.BatchNormalization()(y2)
    act4 = layers.LeakyReLU(alpha=0.1)(bn4)
    dp4 = layers.Dropout(0.2)(act4)

    # Concatenation of LSTM and FCN paths
    z = layers.concatenate([y, dp4])

    # Reshape to do 1D conv for feature sparsification
    z1 = layers.Reshape((192, 1))(z)
    z2 = layers.Conv1D(64, 5, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(z1)
    bn5 = layers.BatchNormalization()(z2)
    act5 = layers.LeakyReLU(alpha=0.1)(bn5)
    dp5 = layers.Dropout(0.2)(act5)
    z3 = layers.Conv1D(32, 3, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp5)
    bn6 = layers.BatchNormalization()(z3)
    act6 = layers.LeakyReLU(alpha=0.1)(bn6)
    dp6 = layers.Dropout(0.2)(act6)
    z4 = layers.GlobalAveragePooling1D()(dp6)
    out = layers.Dense(2, activation='softmax')(z4)

    model = Model(inputs=[sig_input, sig_input2], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_model21(X1, X2):
    """
    Dense layer after 2 Conv1D layers in the fusion network
    :param X1: Time domain features
    :param X2: PSD features
    :return: model
    """
    sig_input = layers.Input(shape=(X1.shape[1], X1.shape[2]))
    sig_input2 = layers.Input(shape=(X2.shape[1],))
    # LSTM path of the network
    x = layers.LSTM(8)(sig_input)
    dp = layers.Dropout(0.8)(x)

    # Fully convolutional path of the network
    x1 = layers.Conv1D(128, 10, padding='same', kernel_initializer='he_uniform',
                       activity_regularizer=regularizers.l2(0.001))(sig_input)
    bn1 = layers.BatchNormalization()(x1)
    act1 = layers.LeakyReLU(alpha=0.1)(bn1)
    dp1 = layers.Dropout(0.2)(act1)

    x2 = layers.Conv1D(64, 5, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp1)
    bn2 = layers.BatchNormalization()(x2)
    act2 = layers.LeakyReLU(alpha=0.1)(bn2)
    dp2 = layers.Dropout(0.2)(act2)

    x3 = layers.Conv1D(128, 5, padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(0.001))(dp2)
    bn3 = layers.BatchNormalization()(x3)
    act3 = layers.LeakyReLU(alpha=0.1)(bn3)
    dp3 = layers.Dropout(0.2)(act3)

    y = layers.GlobalAveragePooling1D()(dp3)

    # PSD features
    y1 = layers.Dense(64, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(sig_input2)
    bn4 = layers.BatchNormalization()(y1)
    act4 = layers.LeakyReLU(alpha=0.1)(bn4)
    dp4 = layers.Dropout(0.2)(act4)

    # Concatenation of LSTM and FCN paths
    z = layers.concatenate([dp, y, dp4])

    # Reshape to do 1D conv for feature sparsification
    z1 = layers.Reshape((200, 1))(z)
    z2 = layers.Conv1D(64, 5, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(z1)
    bn5 = layers.BatchNormalization()(z2)
    act5 = layers.LeakyReLU(alpha=0.1)(bn5)
    dp5 = layers.Dropout(0.2)(act5)
    z3 = layers.Conv1D(32, 3, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(dp5)
    bn6 = layers.BatchNormalization()(z3)
    act6 = layers.LeakyReLU(alpha=0.1)(bn6)
    dp6 = layers.Dropout(0.2)(act6)
    z5 = layers.GlobalAveragePooling1D()(dp6)
    z6 = layers.Dense(32, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001))(z5)
    bn7 = layers.BatchNormalization()(z6)
    act7 = layers.LeakyReLU(alpha=0.1)(bn7)
    dp7 = layers.Dropout(0.2)(act7)
    out = layers.Dense(2, activation='softmax')(dp7)

    model = Model(inputs=[sig_input, sig_input2], outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
