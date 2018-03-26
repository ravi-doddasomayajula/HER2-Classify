execfile('import_modules.py')
execfile('mk_dir.py')
#execfile('encoderfit.py')
execfile('get_model.py')
execfile('print_model.py')
execfile('epochs.py')
execfile('plot_learning_curve.py')

#execfile('model_save.py')
#execfile('model_save_flip.py')

model = get_model()

history = []
history_flip = []

for ite in range(100):

	history = model.fit(
	    X_train,
	    y_train_dummy,
	    epochs=epochs,
	    verbose=1,	    
	    batch_size=batch_size, 
	    validation_data = (X_test,y_test_dummy))

        model.save_weights('binary_model_' + str(ite) + '.h5' )

	execfile('accuracy.py')

    	plot_learning_curve(history,ite)

	
	history_flip = model.fit(
	    X_train_new,
	    y_train_dummy_new,
	    epochs=epochs,
	    verbose=1,	    
	    batch_size=batch_size, 
	    validation_data = (X_test_new,y_test_dummy_new))

        model.save_weights('binary_model_flip_' + str(ite) + '.h5' )

	execfile('accuracy.py')

	plot_learning_curve(history_flip,ite)



