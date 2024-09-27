## Import Library
from tensorflow.keras.models import Sequential # type: ignore 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout ## type: ignore 
from tensorflow.keras.optimizers import Adam # type: ignore 
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore 

print('TEST Import Library Success')



## ImageDataGenerator (Rescale)
train_data = ImageDataGenerator(rescale=1./255)
validate_data = ImageDataGenerator(rescale=1./255)


## Prepreocessing Data Train
data_train_generator = train_data.flow_from_directory(
    'data_train',
    target_size=(48,48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

## Preprocessing Data Test
data_validate_generator = validate_data.flow_from_directory(
    'data_test',
    target_size=(48,48),
    batch_size=64,
    class_mode='categorical',
    color_mode='grayscale'
)


## Create Model
model = Sequential()


## Convolutional Layer
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

## Flatten Layer
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

## Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001, decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

## Train Model with CNN
model_train = model.fit(
    data_train_generator,
    steps_per_epoch=28709//64,
    # epochs=50,
    epochs=5,
    validation_data=data_validate_generator,
    validation_steps=7178//64
)


## Save Model as Json
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

# Save model to .h5 file
model.save_weights('model.weights.h5')

print("TRAINING SUCCESS")



