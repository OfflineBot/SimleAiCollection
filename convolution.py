import numpy as np

def convolution(image, kernel):
    # Assuming 'image' and 'kernel' are 2D arrays
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    result = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            result[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return result

def max_pooling(feature_map, pool_size):
    # Assuming 'feature_map' is a 2D array
    height, width = feature_map.shape
    pooled_height = height // pool_size
    pooled_width = width // pool_size
    result = np.zeros((pooled_height, pooled_width))

    for i in range(0, pooled_height * pool_size, pool_size):
        for j in range(0, pooled_width * pool_size, pool_size):
            result[i//pool_size, j//pool_size] = np.max(feature_map[i:i+pool_size, j:j+pool_size])

    return result

def kernel_update(kernel, learning_rate, gradient):
    # Assuming 'kernel' and 'gradient' are 2D arrays of the same shape
    updated_kernel = kernel - learning_rate * gradient
    return updated_kernel

# Example usage
image = np.random.rand(6, 6)  # Replace this with your actual image data
kernel = np.random.rand(3, 3)  # Replace this with your actual kernel
learning_rate = 0.01

# Convolution
conv_result = convolution(image, kernel)

# Max pooling
pool_size = 2
pooled_result = max_pooling(conv_result, pool_size)

# Kernel update (assuming you have a gradient)
gradient = np.random.rand(*kernel.shape)  # Replace this with your actual gradient
updated_kernel = kernel_update(kernel, learning_rate, gradient)

print(pooled_result)