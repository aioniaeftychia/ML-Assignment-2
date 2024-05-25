import numpy as np
import matplotlib.pyplot as plt
# Load the dataset using np.genfromtxt

data = np.genfromtxt('traindata.txt', dtype=float, delimiter=',')
labels = np.genfromtxt('trainlabels.txt', dtype=int, delimiter=',')

# Check the shape of the loaded data
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

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
images = features_cleaned.reshape(-1, 32, 32)

# we clean by checking each pixel, and the difference between it and it's neighbours.
# The functions are divided for optimisation, but it also avoids 'out of bounds' errors.
# usually, use the stat library for mean, but the automarker might not have this.
def clean(matrix, e):
    rowEnd = matrix.shape[0]-1
    colEnd = matrix.shape[1]-1
    def mean(arr):
        sum = 0
        for i in arr:
            sum = sum  + i
        sum = sum / len(arr)
        return int(sum)
    def contrasted(x, y, e):
        if abs(x-y) > e:
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
        if contrasted(origin, a, e) and contrasted(origin, b, e) and contrasted(origin, c, e) and contrasted(origin, d, e):
            origin = mean({a, b, c, d})
        return origin

    matrix[0, 0] = corner(matrix[0, 0], matrix[0, 1], matrix[1, 0], e)
    matrix[0, colEnd] = corner(matrix[0, colEnd], matrix[0, colEnd-1], matrix[1, colEnd], e)
    matrix[rowEnd, 0] = corner(matrix[rowEnd, 0], matrix[rowEnd-1, 0], matrix[rowEnd, 1], e)
    matrix[rowEnd, colEnd] = corner(matrix[rowEnd, colEnd], matrix[rowEnd-1, colEnd], matrix[rowEnd, colEnd-1], e)

    for r in range(1, rowEnd-1):
        matrix[r, 0] = side(matrix[r, 0], matrix[r-1, 0], matrix[r, 1], matrix[r+1, 0], e)
    for r in range(1, rowEnd-1):
        matrix[r, colEnd] = side(matrix[r, colEnd], matrix[r-1, colEnd], matrix[r, colEnd-1], matrix[r+1, colEnd], e)
    
    for c in range(1, rowEnd-1):
        matrix[0, c] = side(matrix[0, c], matrix[0, c-1], matrix[1, c], matrix[0, c+1], e)
    for c in range(1, rowEnd-1):
        matrix[rowEnd, c] = side(matrix[rowEnd, c], matrix[rowEnd, c-1], matrix[rowEnd-1, c], matrix[rowEnd, c+1], e)

    for r in range(1, rowEnd-1):
        for c in range(1, colEnd-1):
            matrix[r, c] = centre(matrix[r, c], matrix[r, c-1], matrix[r, c+1], matrix[r-1, c], matrix[r+1, c], e)

    return matrix

# Verify the shape of the reshaped images array
print("Images shape:", images.shape)
data_point_index = 0
while data_point_index != -1:
    # Choose the data point index to plot
    data_point_index = int(input(f"Enter an index between 0 and {len(images) - 1}: "))

    if data_point_index == -1:
        break

    # Ensure the chosen index is within the valid range
    if data_point_index < 0 or data_point_index >= len(images):
        raise IndexError(f"Index {data_point_index} is out of bounds. Please enter a valid index.")

    current_datapoint = clean(images[data_point_index], 180)
   # current_datapoint = images[data_point_index]
    # Plot the selected image
    plt.imshow(current_datapoint, cmap='gray')
    plt.title(f'Label: {labels[data_point_index]}, Rotation: {rotations[data_point_index]}')
    plt.show()


