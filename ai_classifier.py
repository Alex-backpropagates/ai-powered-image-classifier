import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from PIL import Image
import os
import sys
import pathlib

IMG_SIZE = (64, 64)
BATCH_SIZE = 16

def create_model():
    return keras.Sequential([
        keras.layers.Flatten(input_shape=(64, 64, 3)),
        keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])

def train_model():
    #Get all pictures from the 3 folders
    racine = pathlib.Path('.')

    banana_images = list(racine.glob('banana/*'))
    cherry_images = list(racine.glob('cherry/*'))
    together_images = list(racine.glob('together/*'))
    
    print(f"Banana number: {len(banana_images)}")
    print(f"Cherry number: {len(cherry_images)}")
    print(f"Together number: {len(together_images)}")
    
    if len(banana_images) == 0 or len(cherry_images) == 0 or len(together_images) == 0:
        print("ERROR : There are no pictures")
        return
    
    # Assign 
    image_paths = []
    labels = []
    
    for img_path in banana_images:
        if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            image_paths.append(str(img_path))
            labels.append(0)  # 0 for banana
    
    for img_path in cherry_images:
        if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            image_paths.append(str(img_path))
            labels.append(1)  # 1 for cherry

    for img_path in together_images:
        if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            image_paths.append(str(img_path))
            labels.append(2)  # 2 for together
    
    # Convert picture in inputs for the neural network
    def loadpictureinneural(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(loadpictureinneural)
    dataset = dataset.shuffle(len(image_paths))
    
   
    train_size = len(image_paths)
    train_ds = dataset.take(train_size).batch(BATCH_SIZE)
    val_ds = dataset.skip(train_size).batch(BATCH_SIZE)
    
    # Model
    model = create_model()
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])


    model.fit(train_ds, validation_data=val_ds, epochs=200, callbacks=[early_stopping])
    model.save('fruit_model.h5')
    print("End of Training")
 
def predict_image():
    
    test_dir = "test"

    if not os.path.exists('fruit_model.h5'):
        print("ERROR: no training file 'fruit_model.h5'. Train the model first")
        return
    
    if not os.path.exists(test_dir):
        print("ERROR: folder test not found.")
        return
    
    model = keras.models.load_model('fruit_model.h5')

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255 
    )
    
    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        classes=None  
    )

    class_labels = list(test_generator.class_indices.keys())
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    class_names = ['banana', 'cherry', 'together']
    #predicted_classes = class_names[np.argmax(predictions, axis=1)]
    confidence = np.max(predictions)
    print(f"Guesses: {predicted_classes} ({confidence:.2%})")

    #ACCURACY
    true_classes = test_generator.classes
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"Accuracy: {accuracy:.4f}")

    #MATRIX CONFUSION
    cm = confusion_matrix(true_classes, predicted_classes)
    print(pd.DataFrame(cm, columns=class_labels, index=class_labels))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py [train|predict]")
    elif sys.argv[1] == "train":
        train_model()
    elif sys.argv[1] == "predict":
        predict_image()
    else:
        print("Invalid command. Use 'train' or 'predict'")