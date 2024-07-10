import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Path to directories
train_data_path = r"C:\Users\Gaurav\Downloads\Dataset\train"
test_data_path = r"C:\Users\Gaurav\Downloads\Dataset\test"
model_path = "C:\\Users\\Gaurav\\Downloads\\Dataset\\cnn\\hen_disease_detection.keras"

# Function to load images and filter out corrupted ones
def load_images_from_directory(directory):
    valid_images = []
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                img = cv2.imread(os.path.join(root, file))
                if img is not None:
                    label = int("hen" in root)
                    valid_images.append((os.path.join(root, file), label))
            except Exception as e:
                print(f"Error processing {file}: {e}")
    return valid_images

# Load images and filter out corrupted ones
train_images = load_images_from_directory(train_data_path)
test_images = load_images_from_directory(test_data_path)

# Debugging: Print the first few entries of train_images and test_images
print("Train Images:")
print(train_images[:5])
print("Test Images:")
print(test_images[:5])

# Shuffle the datasets
np.random.shuffle(train_images)
np.random.shuffle(test_images)

# Define a function to load and preprocess images
def load_and_preprocess_image(image_path, label):
    img = tf.io.read_file(image_path)  # Read image file
    img = tf.image.decode_jpeg(img, channels=3)  # Decode JPEG image to tensor with 3 color channels
    img = tf.image.resize(img, [150, 150])  # Resize image to 150x150 pixels
    img = tf.cast(img, tf.float32) / 255.0  # Cast image to float32 and normalize pixel values to [0, 1]
    return img, label

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(train_images)  # Create dataset from list of image paths and labels
train_dataset = train_dataset.map(load_and_preprocess_image)  # Apply preprocessing to each image
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(16)  # Shuffle and batch the dataset

test_dataset = tf.data.Dataset.from_tensor_slices(test_images)  # Create dataset from list of image paths and labels
test_dataset = test_dataset.map(load_and_preprocess_image)  # Apply preprocessing to each image
test_dataset = test_dataset.batch(16)  # Batch the dataset

# Model Architecture
cnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=[150,150,3]),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=256, kernel_size=3),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile Model
cnn_model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Train Model
try:
    history = cnn_model.fit(
        train_dataset,
        steps_per_epoch=len(train_dataset),
        epochs=100,
        validation_data=test_dataset,
        validation_steps=len(test_dataset),
        callbacks=callbacks_list
    )
except OSError as e:
    print("Error:", e)
    print("Skipping current batch...")
    pass

# Plotting
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Function to perform prediction on a single image
def predict_image(img):
    img = cv2.resize(img, (150, 150))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = cnn_model.predict(img)
    return prediction

# Initialize camera
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    prediction = predict_image(frame)

    if prediction[0][0] > 0.5:
        print("Hen detected")
    else:
        print("No hen detected")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Debugging: Print the first few entries of train_images and test_images
print("Train Images:")
print(train_images[:5])
print("Test Images:")
print(test_images[:5])

# Check data types of train_images and test_images
print("Data types:")
print("Train Images - Path:", type(train_images[0][0]), "Label:", type(train_images[0][1]))
print("Test Images - Path:", type(test_images[0][0]), "Label:", type(test_images[0][1]))
