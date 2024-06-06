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


def clean(matrix, e):
    rowEnd = matrix.shape[0] - 1
    colEnd = matrix.shape[1] - 1

    def mean(arr):
        sum = 0
        for i in arr:
            sum = sum + i
        sum = sum / len(arr)
        return int(sum)

    def contrasted(x, y, e):
        if abs(x - y) > e:
            return True
        else:
            return False

    def corner(origin, a, b, e):
        if contrasted(origin, a, e) and contrasted(origin, b, e):
            origin = mean({a, b})
        return origin

    def side(origin, a, b, c, e):
        if contrasted(origin, a, e) and contrasted(origin, b, e) and contrasted(origin, c, e):
            origin = mean({a, b, c})
        return origin

    def centre(origin, a, b, c, d, e):
        if contrasted(origin, a, e) and contrasted(origin, b, e) and contrasted(origin, c, e) and contrasted(origin, d,
                                                                                                             e):
            origin = mean({a, b, c, d})
        return origin

    matrix[0, 0] = corner(matrix[0, 0], matrix[0, 1], matrix[1, 0], e)
    matrix[0, colEnd] = corner(matrix[0, colEnd], matrix[0, colEnd - 1], matrix[1, colEnd], e)
    matrix[rowEnd, 0] = corner(matrix[rowEnd, 0], matrix[rowEnd - 1, 0], matrix[rowEnd, 1], e)
    matrix[rowEnd, colEnd] = corner(matrix[rowEnd, colEnd], matrix[rowEnd - 1, colEnd], matrix[rowEnd, colEnd - 1], e)

    for r in range(1, rowEnd):
        matrix[r, 0] = side(matrix[r, 0], matrix[r - 1, 0], matrix[r, 1], matrix[r + 1, 0], e)
    for r in range(1, rowEnd):
        matrix[r, colEnd] = side(matrix[r, colEnd], matrix[r - 1, colEnd], matrix[r, colEnd - 1], matrix[r + 1, colEnd],
                                 e)

    for c in range(1, rowEnd):
        matrix[0, c] = side(matrix[0, c], matrix[0, c - 1], matrix[1, c], matrix[0, c + 1], e)
    for c in range(1, rowEnd):
        matrix[rowEnd, c] = side(matrix[rowEnd, c], matrix[rowEnd, c - 1], matrix[rowEnd - 1, c], matrix[rowEnd, c + 1],
                                 e)

    for r in range(1, rowEnd):
        for c in range(1, colEnd):
            matrix[r, c] = centre(matrix[r, c], matrix[r, c - 1], matrix[r, c + 1], matrix[r - 1, c], matrix[r + 1, c],
                                  e)

    return matrix


def darken(matrix, e):
    row = matrix.shape[0]
    col = matrix.shape[1]
    for r in range(row):
        for c in range(col):
            if matrix[r, c] < e:
                matrix[r, c] = 10
    return matrix


def full_cleanup(matrix):
    for point in matrix:
        point = clean(point, 160)
        point = darken(point, 90)
        point = clean(point, 50)
    return images


images = full_cleanup(images)

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
