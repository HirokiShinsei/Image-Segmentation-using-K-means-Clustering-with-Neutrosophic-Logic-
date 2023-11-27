import cv2
import numpy as np
import matplotlib.pyplot as plt

def neutrosophic_mean(image, window_size):
    padding = window_size // 2
    padded_image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)

    mean_image = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + 2 * window_size + 1, j:j + 2 * window_size + 1]
            mean_image[i, j] = np.mean(window)

    return mean_image


def neutrosophic_falsity(mean_image):
    falsity_image = 1 - mean_image
    return falsity_image

def neutrosophic_indeterminacy(mean_image, std_image, diff_image):
    indeterminacy_image = (std_image * diff_image) / np.maximum(std_image, diff_image)
    return indeterminacy_image

def neutrosophic_clustering(truth_image, falsity_image, indeterminacy_image, k):
    h, w = truth_image.shape
    data = np.column_stack((truth_image.flatten(), falsity_image.flatten(), indeterminacy_image.flatten()))

    centroids = np.array([data[np.random.randint(data.shape[0])] for _ in range(k)], dtype=np.float32)

    for _ in range(10):
        distances = np.linalg.norm(data[:, None] - centroids, axis=2)
        cluster_assignments = np.argmin(distances, axis=1)

        for cluster_index in range(k):
            cluster_pixels = data[cluster_assignments == cluster_index]
            if len(cluster_pixels) > 0:
                new_centroid = np.mean(cluster_pixels, axis=0)
                centroids[cluster_index] = new_centroid

    return cluster_assignments.reshape((h, w))

def refine_clusters(cluster_assignments, min_cluster_size):

    # Initialize the refined cluster assignments.
    refined_cluster_assignments = np.copy(cluster_assignments)

    # Iterate over the clusters.
    for cluster_index in range(np.max(cluster_assignments) + 1):
        # Get the pixels in the cluster.
        cluster_pixels = np.where(cluster_assignments == cluster_index)

        # If the cluster has less than the minimum cluster size, merge it with the neighboring cluster with the most overlapping pixels.
        if len(cluster_pixels[0]) < min_cluster_size:
            # Get the neighboring clusters.
            neighboring_clusters = []
            for pixel in cluster_pixels:
                # Get the neighbors of the pixel.
                neighbors = []
                for neighbor_x in range(pixel[0] - 1, pixel[0] + 2):
                    for neighbor_y in range(pixel[1] - 1, pixel[1] + 2):
                        # Check if the neighbor pixel is in the image and not in the current cluster.
                        if (
                            0 <= neighbor_x < cluster_assignments.shape[0]
                            and 0 <= neighbor_y < cluster_assignments.shape[1]
                            and cluster_assignments[neighbor_x, neighbor_y] != cluster_index
                        ):
                            neighbors.append(cluster_assignments[neighbor_x, neighbor_y])

                # Add the neighboring clusters to the list of neighboring clusters.
                neighboring_clusters.extend(neighbors)

            if neighboring_clusters:
                # Get the neighboring cluster with the most overlapping pixels.
                most_overlapping_cluster = np.max(np.bincount(neighboring_clusters))

                # Merge the current cluster with the most overlapping cluster.
                refined_cluster_assignments[cluster_pixels] = most_overlapping_cluster

    return refined_cluster_assignments

def segment_image(refined_cluster_assignments):
    segmented_image = np.zeros_like(refined_cluster_assignments, dtype=np.uint8)

    for cluster_index in range(1, np.max(refined_cluster_assignments) + 1):
        cluster_pixels = np.where(refined_cluster_assignments == cluster_index)
        if len(cluster_pixels[0]) > 0:  # Check if the cluster is not empty
            segmented_image[cluster_pixels] = 255  # Set the pixels corresponding to the cluster to white

    # Invert the colors
    segmented_image = cv2.bitwise_not(segmented_image)

    return segmented_image

def neutrosophic_kmeans(image_path, k, window_size, min_cluster_size):
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    mean_image = neutrosophic_mean(gray_image, window_size)
    falsity_image = neutrosophic_falsity(mean_image)

    std_image = np.std(gray_image)
    diff_image = np.abs(gray_image - mean_image)
    indeterminacy_image = neutrosophic_indeterminacy(mean_image, std_image, diff_image)

    cluster_assignments = neutrosophic_clustering(mean_image, falsity_image, indeterminacy_image, k)
    refined_cluster_assignments = refine_clusters(cluster_assignments, min_cluster_size)
    segmented_image = segment_image(refined_cluster_assignments)

    return segmented_image

def show_results(original, segmented):
    plt.figure(figsize=(8, 4))

    # Original Image
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    # Resulting Segmented Image in Black and White
    plt.subplot(122)
    plt.imshow(segmented, cmap='gray')
    plt.title('Resulting Segmented Image')

    plt.tight_layout()
    plt.show()

# Example usage:
image_path = '2nd_year/Intro to AI/midterm/Lisa.jpg'
k = 3
window_size = 3
min_cluster_size = 160

# Perform neutrosophic k-means clustering
resulting_segmented_image = neutrosophic_kmeans(image_path, k, window_size, min_cluster_size)

# Load the original image for visualization
original_image = cv2.imread(image_path)

# Show the results
show_results(original_image, resulting_segmented_image)