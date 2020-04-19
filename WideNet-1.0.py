
get_ipython().run_line_magic('run', 'Setup.ipynb')
get_ipython().run_line_magic('run', 'ExtraFunctions.ipynb')
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_lstm1 = Bidirectional(LSTM(8,return_sequences=True))(embedded_sequences)


inception = [] 

l_conv_7 = Conv1D(filters=24,kernel_size=7,activation='relu',kernel_regularizer=regularizers.l2(0.01))(l_lstm1)

inception.append(l_conv_7)

l_pool_i1 = MaxPooling1D(3)(l_lstm1)
l_conv_1 = Conv1D(filters=24,kernel_size=1,
                    activation='relu',kernel_regularizer=regularizers.l2(0.01))(l_pool_i1)
inception.append(l_conv_1)

l_merge = Concatenate(axis=1)(inception)
l_pool = MaxPooling1D(4)(l_merge)
l_drop = Dropout(0.5)(l_pool)
l_flat = Flatten()(l_drop)
l_dense = Dense(12, activation='relu')(l_flat)
preds = Dense(7, activation='softmax')(l_dense)

model = Model(sequence_input, preds)
adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
lr_metric = get_lr_metric(adadelta)
model.compile(loss='categorical_crossentropy',
              optimizer=adadelta,
              metrics=['acc'])


get_ipython().system('rm -r logs')


tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=16, write_grads=True , write_graph=True)
model_checkpoints = callbacks.ModelCheckpoint("checkpoint.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
lr_schedule = callbacks.LearningRateScheduler(initial_boost)


model.summary()
model.save('EMO.h5')

print("Training Progress:")
model_log = model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=200, batch_size=50,
          callbacks=[tensorboard, lr_schedule])

import pandas as pd
model.save('WideNet.h5')
pd.DataFrame(model_log.history).to_csv("history-inception.csv")

