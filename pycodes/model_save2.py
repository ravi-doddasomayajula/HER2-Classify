model = get_model()

history_new = model.fit(
    X_train_new,
    y_train_dummy_new,
    epochs=epochs,
    verbose=1,	    
    batch_size=batch_size, 
    validation_data = (X_test_new,y_test_dummy_new))

model.save_weights('binary_model2.h5')
