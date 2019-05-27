import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Input, Sequential, Model
from keras.layers import Dense, LSTM, Masking, Bidirectional, concatenate, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, scale, MinMaxScaler, normalize
from multiprocessing import Pool
import matplotlib.pyplot as plt


def remove_small_classes(df, min):
    # TODO TEM DE SER FIXED PORQUE ISTO AGORA ESTA POR CAMADAS E NAO PERFIS
    uniques = df.cwrb_reference_soil_group.unique()
    for u in uniques:
        cnt = df[df.cwrb_reference_soil_group == u].shape[0]
        if cnt < min:
            df = df[df.cwrb_reference_soil_group != u]
            print('Deleting {} with {} occurrences'.format(u, cnt))

    return df


def treat_data_2(data):
    # Treat Data
    profile_data = data.iloc[:, [11, 12, -1]]
    layer_data = data.drop(data.columns[[0, 1, 11, 12, -1]], axis=1)

    # Treat Profiles
    total_profiles = []
    for row in profile_data.itertuples(index=False):
        total_profiles.append(row[:])

    # Treat Layers
    total_layers = []
    for row in layer_data.itertuples(index=False):
        i = 0
        layers = []
        for j in range(9, 28, 9):
            layers.append(row[i:j])
            i = j
        total_layers.append(layers)

    return np.array(total_profiles), np.array(total_layers)


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

    data_scaled -= data_scaled.mean(axis=0)
    data_scaled /= data_scaled.std(axis=0)
    #data_scaled = normalize(data_scaled)

    data = pd.DataFrame(data=data_scaled, columns=list(
        data.drop(['cwrb_reference_soil_group', 'profile_id'], axis=1).columns))
    data = data.join(data_not_to_scale)
    return data


def create_model(profile_data, layer_data, n_classes):
    input_profile = Input(
        shape=(profile_data.shape[1:]))
    output_profile = Dense(32, activation="relu")(input_profile)

    input_layer = Input(shape=(layer_data.shape[1:]))
    masking_layer = Masking(mask_value=0)(input_layer)
    middle_layer = Bidirectional(LSTM(20))(masking_layer)
    #dropout_layer = Dropout(0.2)(middle_layer)

    join_layer = concatenate([output_profile, middle_layer])

    output_final = Dense(n_classes, activation='softmax')(join_layer)

    opt = Adam(lr=0.001, decay=1e-6)
    model = Model(inputs=[input_profile, input_layer], outputs=output_final)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def get_data():
    inputfile = '../data/mexico_k_1.csv'
    profile_file = '../data/profiles.csv'
    profiles_file = pd.read_csv(profile_file)
    profiles_file = profiles_file[['profile_id', 'cwrb_reference_soil_group']]
    data = pd.read_csv(inputfile)
    data = profiles_file.merge(data, how="inner", left_on=[
        'profile_id'], right_on=['profile_id'])

    data = scale_data(data)
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

    data.fillna(0, inplace=True)
    data = data.drop('country_id', axis=1)

    data = scale_data(data)
    #data = remove_small_classes(data, 15)
    profile_data, layer_data, y = treat_data(data)

    # Treat Labels
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    dummy_y = np_utils.to_categorical(encoded_Y)

    return profile_data, layer_data, dummy_y


# Read Data
profile_data, layer_data, y = get_data_all()


# ACTUAL MODEL
model = create_model(profile_data, layer_data, y.shape[1])


history = model.fit([profile_data, layer_data],
                    y, epochs=40, validation_split=0.1)


plot_loss(history)
plot_accuracy(history)


"""
X = X.drop(columns=list(
    X.loc[:, X.columns.str.contains('profile_layer_id')]))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)


features_contexto = features_contexto.reshape(
    features_contexto.shape[0], 3)
lista_de_perfis = lista_de_perfis.reshape(lista_de_perfis.shape[0], 3 * 9)
# Model
features_contexto, lista_de_perfis = treat_data_2(data)


# Para features contexto
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(3,)))
model.add(Dense(22, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])


X_train, X_test, y_train, y_test = train_test_split(
    features_contexto, dummy_y, test_size=0.15)
model.fit(x=X_train, y=y_train, epochs=30)


# Para lista de perfis
model = Sequential()
model.add(Bidirectional(LSTM(128, activation='relu',  return_sequences=True,
                             input_shape=(lista_de_perfis.shape[1:]))))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Bidirectional(LSTM(128, activation='relu',  return_sequences=True)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Bidirectional(LSTM(128, activation='relu')))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(22, activation='softmax'))



model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])


X_train, X_test, y_train, y_test = train_test_split(
    lista_de_perfis, dummy_y, test_size=0.15)

model.fit(x=X_train, y=y_train, epochs=30, batch_size=128,
          validation_data=(X_test, y_test))


# Test model

test_loss, test_acc = model.evaluate(X_test, y_test)
print('test_acc: ', test_acc)
"""
