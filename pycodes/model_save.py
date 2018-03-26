model = get_model()

# fits the model on batches
#history = model.fit(
#    X_train,
#    y_train,
#    validation_split=0.2,
#    epochs=epochs,
#    shuffle=True,
#    batch_size=batch_size)

history = model.fit(
    X_train,
    y_train_dummy,
    epochs=epochs,
    verbose=1,	    
    batch_size=batch_size, 
    validation_data = (X_test,y_test_dummy))

model.save_weights('binary_model.h5')
