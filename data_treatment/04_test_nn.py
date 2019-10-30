

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from keras.models import Input, Sequential, Model
from keras.layers import Dense, LSTM, TimeDistributed, Masking, Bidirectional, concatenate, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import cohen_kappa_score
from keras.utils import plot_model

import sys
sys.path.append('./utils')
#from nested_lstm import NestedLSTM
#from clr_callback import CyclicLR
#from multiplicative_lstm import MultiplicativeLSTM


def remove_small_classes(df, min):
    # TODO TEM DE SER FIXED PORQUE ISTO AGORA ESTA POR CAMADAS E NAO PERFIS
    uniques = df.cwrb_reference_soil_group.unique()
    for u in uniques:
        cnt = df[df.cwrb_reference_soil_group == u].shape[0]
        if cnt < min:
            df = df[df.cwrb_reference_soil_group != u]
            print('Deleting {} with {} occurrences'.format(u, cnt))

    return df


def get_data_structured():
    inputfile = '../data/test/mexico_k_1_layers_3.csv'
    profile_file = '../data/profiles.csv'
    profiles_file = pd.read_csv(profile_file)
    profiles_file = profiles_file[['profile_id', 'cwrb_reference_soil_group']]
    data = pd.read_csv(inputfile)
    data = profiles_file.merge(data, how="inner", left_on=[
        'profile_id'], right_on=['profile_id'])

    #data = scale_data(data)
    #data = remove_small_classes(data, 15)
    profile_data, layer_data, y = treat_data_structured(data)

    # Treat Labels
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    dummy_y = np_utils.to_categorical(encoded_Y)

    return profile_data, layer_data, dummy_y


def treat_data_structured(data):
    # Treat Data
    profile_data = data[['latitude', 'longitude']]
    layer_data = data.drop([
                           'latitude', 'longitude', 'cwrb_reference_soil_group', 'profile_id', 'n_layers'], axis=1)
    y = data.cwrb_reference_soil_group

    # Treat Profiles
    total_profiles = []
    for row in profile_data.itertuples(index=False):
        total_profiles.append(row[:])

    # Treat Layers
    total_layers = []
    for row in layer_data.itertuples(index=False):
        i = 0
        layers = []
        for j in range(9, 29, 9):
            layers.append(row[i:j])
            i = j
        total_layers.append(layers)

    return np.array(total_profiles), np.array(total_layers), y


def plot_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_accuracy(history):
    plt.clf()
    acc = history.history['acc']
    epochs = range(1, len(acc) + 1)
    val_acc = history.history['val_acc']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def treat_data(data):
    profile_data = []
    layer_data = []
    profile_ids = data.profile_id.unique()
    max_n_layers = data.profile_id.value_counts().head(1).values[0]
    # All except latitude longitude profile id and class
    features_per_layer = len(data.columns) - 4
    y = []

    for i, id in enumerate(profile_ids):
        if i % 100 == 0:
            print(i)

        # Find the layers for this profile and sort them
        layers = data[data['profile_id'] == id]
        layers = layers.sort_values(by=['lower_depth'])

        # Add the layer, filling the missing layers with 0
        layer = np.zeros([max_n_layers, features_per_layer])
        for j, l in enumerate(layers.drop(['profile_id', 'cwrb_reference_soil_group', 'latitude', 'longitude'], axis=1).values):
            layer[j] = l
        layer_data.append(layer)

        # Fill the profile data
        profile = layers.iloc[0][['latitude', 'longitude']]
        profile_data.append(profile.values)

        y.append(layers.iloc[0].cwrb_reference_soil_group)

    return np.array(profile_data).reshape(len(profile_data), 2), np.array(layer_data), y


def scale_data(data):
    data_not_to_scale = data[['cwrb_reference_soil_group', 'profile_id']]

    data_scaled = data.drop(
        ['cwrb_reference_soil_group', 'profile_id'], axis=1).values

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_scaled)

    #data_scaled -= data_scaled.mean(axis=0)
    #data_scaled /= data_scaled.std(axis=0)

    data = pd.DataFrame(data=data_scaled, columns=list(
        data.drop(['cwrb_reference_soil_group', 'profile_id'], axis=1).columns))
    data = data.join(data_not_to_scale)
    return data


def get_data():
    inputfile = '../data/mexico_k_1.csv'
    profile_file = '../data/profiles.csv'
    profiles_file = pd.read_csv(profile_file)
    profiles_file = profiles_file[['profile_id', 'cwrb_reference_soil_group']]
    data = pd.read_csv(inputfile)
    data = profiles_file.merge(data, how="inner", left_on=[
        'profile_id'], right_on=['profile_id'])

    data.fillna(-1., inplace=True)
    #data = scale_data(data)
    data = remove_small_classes(data, 15)
    profile_data, layer_data, y = treat_data(data)

    # Treat Labels
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    dummy_y = np_utils.to_categorical(encoded_Y)

    return profile_data, layer_data, dummy_y


def get_data_all():
    inputfile = '../data/all.csv'
    profile_file = '../data/profiles.csv'
    profiles_file = pd.read_csv(profile_file)
    profiles_file = profiles_file[['profile_id', 'cwrb_reference_soil_group']]
    data = pd.read_csv(inputfile)
    data = profiles_file.merge(data, how="inner", left_on=[
        'profile_id'], right_on=['profile_id'])

    data.fillna(-1., inplace=True)
    data = data.drop('country_id', axis=1)

    #data = scale_data(data)
    #data = remove_small_classes(data, 15)
    profile_data, layer_data, y = treat_data(data)

    # Treat Labels
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    dummy_y = np_utils.to_categorical(encoded_Y)

    return profile_data, layer_data, dummy_y


def create_model_linear(profile_data, layer_data, n_classes):
    model = Sequential()
    model.add(Dense(64, activation="relu",
                    input_shape=(layer_data.shape[1:])))

    model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax'))

    opt = Adam(lr=0.00001, decay=1e-6)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def create_model(profile_data, layer_data, n_classes):
    input_profile = Input(
        shape=(profile_data.shape[1:]))
    output_profile = Dense(32, activation="relu")(input_profile)

    input_layer = Input(shape=(layer_data.shape[1:]))
    masking_layer = Masking(mask_value=0.0)(input_layer)
    middle_layer = Bidirectional(
        LSTM(16, return_sequences=True))(masking_layer)

    dropout_layer = TimeDistributed(Dropout(0.2))(middle_layer)

    after_dropout_layer = Bidirectional(LSTM(16))(dropout_layer)
    #after_dropout_layer = Bidirectional(NestedLSTM(units=64, depth=2))(dropout_layer)
    #after_dropout_layer = Bidirectional(MultiplicativeLSTM(16))(dropout_layer)

    join_layer = concatenate([output_profile, after_dropout_layer])

    test = Dense(16, activation="relu")(join_layer)

    output_final = Dense(n_classes, activation='softmax')(test)

    opt = Adam(lr=0.0005, decay=1e-6)
    model = Model(inputs=[input_profile, input_layer], outputs=output_final)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Read Data
profile_data, layer_data, y = get_data()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# Cyclic learning rate test
# clr = CyclicLR(base_lr=0.0001, max_lr=0.001, step_size=(
#    (3) * (profile_data.shape[0])))
# Cyclic test

# ACTUAL MODEL
model = create_model(profile_data, layer_data, y.shape[1])

X_train_profile = profile_data[1000:]
X_train_layer = layer_data[1000:]
X_test_layer = layer_data[:1000]
X_test_profile = profile_data[:1000]
y_train = y[1000:]
y_test = y[:1000]

plot_model(model, show_layer_names=False, to_file='model.pdf')


history = model.fit([X_train_profile, X_train_layer],
                    y_train, epochs=500, validation_split=0.2, callbacks=[es])


plot_loss(history)
plot_accuracy(history)


y_pred = model.predict([X_test_profile, X_test_layer])


print(
    f'kappa {cohen_kappa_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))}')
