import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pandas as pd
from nptdms import TdmsFile

# Git_Folder = os.path.dirname(os.path.dirname(os.getcwd()))
# Christian_folder = os.path.join(os.path.dirname(Git_Folder))
# Fig_folder = os.path.join(Git_Folder, 'Figures')

def create_dataset(
        # data_file: str,    
        seq_length: int, #5000
        X: np.ndarray, #np.array,
        Y: np.ndarray, #np.array,
        # Columns_X: str,
        # Columns_Y: str
):
    # if data_file.endswith('.csv'):
    #     Data = pd.read_csv(data_file)
    # elif data_file.endswith('.tdms'):
    #     tdms_file = TdmsFile.read(data_file)
    #     Data = tdms_file.as_dataframe()

    # X = np.array(Data[Columns_X])
    # Y = np.array(Data[Columns_Y])

    DATASET_SIZE  = int(X.shape[0] / seq_length)

    mean_x = np.mean(X)
    std_x = np.std(X)
    X = (X-mean_x)/std_x

    X = np.expand_dims(X, axis = 1)
    Y = np.expand_dims(Y, axis = 1)

    dataset = tf.data.Dataset.from_tensor_slices(((X, tf.zeros_like(Y)), Y))
    dataset = dataset.batch(seq_length, drop_remainder=True)

    train_size = int(0.7 * DATASET_SIZE)
    val_size = int(0.15 * DATASET_SIZE)
    test_size = int(0.15 * DATASET_SIZE)

    full_dataset = dataset #tf.data.TFRecordDataset(FLAGS.input_file)
    full_dataset = full_dataset.shuffle(buffer_size=DATASET_SIZE)
    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)

    return train_dataset, test_dataset, val_dataset

import keras
from keras_hub.layers import TransformerEncoder, TransformerDecoder
def def_model(
        EMBED_DIM = 64,          # Projection dimension for time-series features
        INTERMEDIATE_DIM = 256, #128  # Transformer feedforward dimension
        NUM_HEADS = 4,           # Number of attention heads
        ENC_TIMESTEPS = 5000,     # Input sequence length (adjust to your data)
        DEC_TIMESTEPS = 5000    # Output sequence length (adjust to your data)
):
    class PositionalEncoding(keras.layers.Layer):
        def __init__(self, sequence_length, output_dim, **kwargs):
            super().__init__(**kwargs)
            self.position_embedding = keras.layers.Embedding(
                input_dim=sequence_length, output_dim=output_dim
            )
        
        def call(self, inputs):
            positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
            position_embeddings = self.position_embedding(positions)
            return inputs + position_embeddings
        
    
    # Encoder Architecture
    encoder_inputs = keras.Input(shape=(ENC_TIMESTEPS, 1))
    x = keras.layers.Dense(EMBED_DIM)(encoder_inputs)  # Project to transformer dim
    x = PositionalEncoding(ENC_TIMESTEPS, EMBED_DIM)(x)
    encoder_outputs = TransformerEncoder(
        intermediate_dim=INTERMEDIATE_DIM,
        num_heads=NUM_HEADS,
        activation = "gelu",
        layer_norm_epsilon = 1e-3,
        dropout = 0.1
    )(x)

    # Decoder Architecture
    decoder_inputs = keras.Input(shape=(DEC_TIMESTEPS, 1))
    x = keras.layers.Dense(EMBED_DIM)(decoder_inputs)  # Project to transformer dim
    x = PositionalEncoding(DEC_TIMESTEPS, EMBED_DIM)(x)
    decoder_outputs = TransformerDecoder(
        intermediate_dim=INTERMEDIATE_DIM,
        num_heads=NUM_HEADS,
        activation = "gelu",
        layer_norm_epsilon = 1e-3,
        dropout = 0.1

    )(
        decoder_sequence=x,
        encoder_sequence=encoder_outputs  # Cross-attention connection
    )
    decoder_outputs = keras.layers.Dense(1, activation="linear")(decoder_outputs)

    # Full model
    transformer = keras.Model(
        [encoder_inputs, decoder_inputs],
        decoder_outputs,
        name="time_series_transformer"
    )
    return transformer

import pickle

def train_model(
        initial_learning_rate = 1e-5,
        warmup_steps = 1000,
        target_learning_rate = 1e-4,
        decay_steps = 5000*3,
        batch_size = 128,
        epochs = 100,
        save_path = "Transformer_models",
        train_dataset = tf.data.Dataset,
        val_dataset = tf.data.Dataset,
        hist_path = "test.csv",
        plt_path = "test.svg",
        type_model = "test",
        transformer = def_model()


):
    # dataset, test_dataset = create_dataset(**dataset_args)

    # lr_warmup_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate, decay_steps, warmup_target=target_learning_rate,
    #     warmup_steps=warmup_steps
    # )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=target_learning_rate)#lr_warmup_decayed_fn)

    transformer.compile(
        optimizer=optimizer, loss="mse"
    )

    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=0.001)

    history = transformer.fit(
        train_dataset,
        batch_size=batch_size,
        epochs=epochs,
        # validation_split=0.2,
        validation_data=val_dataset,
        callbacks=[callback],
        verbose = 2
    )
    # Save model
    transformer.save(save_path)
    
    # file = open(pickle_path, 'wb')
    # pickle.dump(history, file)
    # file.close()
    pd.DataFrame(history.history).to_csv(hist_path)


    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(f"Loss for {type_model}")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(plt_path)
    plt.show()

def train_lots_models(tdms_file, Git_Folder, dataset_folder):
    Data = TdmsFile.read(tdms_file).as_dataframe()

    cols = list(Data.columns)
    X_cols = [x for x in cols if x.endswith('ai0\'')]
    Y_cols = [x for x in cols if x.endswith('ai1\'')]
    type_data = [x.split('10s')[1].split('\'')[0] for x in X_cols]

    for i in range(len(X_cols)):
        train_dataset, test_dataset, val_dataset = create_dataset(
            seq_length = 5000, X = Data[X_cols[i]].dropna().to_numpy(), Y = Data[Y_cols[i]].dropna().to_numpy()
        )
        train_dataset.save(os.path.join(dataset_folder, f"many_train_{type_data[i]}_{os.path.split(tdms_file)[-1].split("_2025")[0]}"))
        test_dataset.save(os.path.join(dataset_folder, f"many_test_{type_data[i]}_{os.path.split(tdms_file)[-1].split("_2025")[0]}"))
        val_dataset.save(os.path.join(dataset_folder, f"many_val_{type_data[i]}_{os.path.split(tdms_file)[-1].split("_2025")[0]}"))
        model_name = type_data[i] + "_" + os.path.split(tdms_file)[-1].split("_2025")[0]
        try:
            train_model(
                train_dataset = train_dataset,
                val_dataset = val_dataset,
                save_path = os.path.join(Git_Folder, "Python", 'Atomspc', "Transformer_models", f"{type_data[i]}_{os.path.split(tdms_file)[-1].split("_2025")[0]}.keras"),
                pickle_path = os.path.join(dataset_folder, f"history_{type_data[i]}_{os.path.split(tdms_file)[-1].split("_2025")[0]}.pkl"),
                plt_path = os.path.join(Git_Folder, "Python", 'Atomspc', "Training_hist", f"loss_hist_{type_data[i]}_{os.path.split(tdms_file)[-1].split("_2025")[0]}.svg"),
                type_model = model_name
            )
        except NameError:
            print("Gone wrong with:", model_name)

# def test_model():