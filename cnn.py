import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import regularizers

# Ensure that the correct versions of libraries are installed
print(f'numpy version: {np.__version__}')
print(f'tensorflow version: {tf.__version__}')

# Load the dataset using np.genfromtxt
data = np.genfromtxt('traindata.txt', dtype=float, delimiter=',')
labels = np.genfromtxt('trainlabels.txt', dtype=int, delimiter=',')

# Separate features and rotation values
features = data[:, :-1]
rotations = data[:, -1]

# Identify columns with negative values (assumed noise)
noise_columns = [col for col in range(features.shape[1]) if (features[:, col] < 0).any()]

# Identify columns immediately after the noise columns
adjusted_columns = [col + 1 for col in noise_columns if col + 1 < features.shape[1]]

# Divide values in the adjusted columns by 10
features[:, adjusted_columns] = features[:, adjusted_columns] / 10

# Drop the noise columns
features_cleaned = np.delete(features, noise_columns, axis=1)

# Verify the new shape after removing noise columns
print("Cleaned features shape:", features_cleaned.shape)

# Check if we have 1024 columns after removing noise and rotation column
if features_cleaned.shape[1] != 1024:
    raise ValueError("Unexpected number of columns after cleaning. Expected 1024 columns.")

# Reshape each row into a 32x32 image
images = features_cleaned.reshape(-1, 32, 32, 1)

# Rotate images based on rotation values
for i in range(len(images)):
    if rotations[i] == 2:
        images[i] = np.rot90(images[i], 2)
    elif rotations[i] == 1:
        images[i] = np.rot90(images[i], 3)
    elif rotations[i] == 3:
        images[i] = np.rot90(images[i], 1)

# Normalize pixel values to be between 0 and 1
images = images / 255.0

# One-hot encode labels
labels = tf.keras.utils.to_categorical(labels, num_classes=21)

# Split data into training, validation, and test sets
train_images, train_labels = images[:4200], labels[:4200]
val_images, val_labels = images[4200:4725], labels[4200:4725]
test_images, test_labels = images[4725:], labels[4725:]

# Shuffle the data
# indices = np.arange(images.shape[0])
# np.random.shuffle(indices)
# images = images[indices]
# labels = labels[indices]

# Split data into training, validation, and test sets
# train_size = int(0.8 * images.shape[0])
# val_size = int(0.1 * images.shape[0])
# test_size = images.shape[0] - train_size - val_size

# train_images, train_labels = images[:train_size], labels[:train_size]
# val_images, val_labels = images[train_size:train_size + val_size], labels[train_size:train_size + val_size]
# test_images, test_labels = images[train_size + val_size:], labels[train_size + val_size:]

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1), kernel_regularizer=regularizers.l2(0.0001)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.07))
model.add(Dense(21, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy:', test_acc)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Predict labels for the test set
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

# Write the true and predicted labels to a text file
with open('test_predictions.txt', 'w') as f:
    f.write('TrueLabel,PredictedLabel\n')
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        f.write(f'{true_label},{predicted_label}\n')
