import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def Train_model(model_name, X_train, Y_train, num_encoder_tokens, num_decoder_tokens, latent_dim, epochs, batch_size):
    # Define an input sequence and process it.
    encoder_inputs = tf.keras.Input(shape=(num_encoder_tokens, 1))
    encoder = tf.keras.layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = tf.keras.Input(shape=(num_decoder_tokens, 1))

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(1, activation="linear")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(
        optimizer="adam", loss="mse"
    )
    history = model.fit(
        [X_train, np.zeros_like(Y_train)],
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
    )
    # Save model
    model.save(model_name + ".keras")
    return history
def PlotPredictions(model_name, X_test, Y_test, test_index_min, test_index_max):

    model = tf.keras.models.load_model(model_name + ".keras")

    test_index = np.arange(test_index_min, test_index_max)

    predictions = model.predict([X_test[test_index_min:test_index_max], np.zeros_like(Y_test[test_index_min:test_index_max])])

    for i in range(test_index_max-test_index_min):
        plt.plot(X_test[test_index[i]][:, 0], label = 'input')
        plt.show()
        plt.plot(predictions[i][:, 0], label = 'prediction', alpha = 0.5)
        plt.plot(Y_test[test_index[i]][:, 0], label = 'target', alpha = 0.5)
        plt.legend()
        plt.show()
