def get_model():
    model = Sequential()
    inputshape = (600,760,3)
    batchsize = 16		
    model.add(Lambda(lambda x: x * 1./255., input_shape=inputshape, output_shape=inputshape))
    model.add(Conv2D(batchsize, (3, 3), input_shape=inputshape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))
    
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(3))
    model.add(Activation('softmax'))	

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                #metrics=['categorical_accuracy'])
		#metrics=['accuracy'])
		metrics=['accuracy', 'binary_accuracy', 'categorical_accuracy'])


    return model

