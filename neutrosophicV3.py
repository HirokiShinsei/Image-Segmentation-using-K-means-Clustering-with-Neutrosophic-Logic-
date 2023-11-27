import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def neutrosophic_kmeans(image_path, k, window_size, min_cluster_size):
    # Read the original image
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Calculate neutrosophic mean and falsity
    mean_image = cv2.blur(gray_image, (window_size, window_size))
    falsity_image = 1 - mean_image

    # Calculate standard deviation and difference image
    std_image = np.std(gray_image)
    diff_image = np.abs(gray_image - mean_image)

    # Calculate indeterminacy image
    indeterminacy_image = (std_image * diff_image) / np.maximum(std_image, diff_image)

    # Stack the features for clustering
    data = np.column_stack((mean_image.flatten(), falsity_image.flatten(), indeterminacy_image.flatten()))

    # Use scikit-learn's KMeans for clustering
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=10, random_state=0)
    cluster_assignments = kmeans.fit_predict(data)

    # Refine clusters
    refined_cluster_assignments = refine_clusters(cluster_assignments, min_cluster_size)

    # Segment the image based on refined clusters
    segmented_image = segment_image(refined_cluster_assignments)

    return segmented_image

def refine_clusters(cluster_assignments, min_cluster_size):
    refined_cluster_assignments = np.copy(cluster_assignments)

    for cluster_index in range(np.max(cluster_assignments) + 1):
        cluster_pixels = np.where(cluster_assignments == cluster_index)

        if len(cluster_pixels[0]) < min_cluster_size:
            neighboring_clusters = []
            for pixel in zip(*cluster_pixels):
                neighbors = [
                    cluster_assignments[i, j]
                    for i in range(pixel[0] - 1, pixel[0] + 2)
                    for j in range(pixel[1] - 1, pixel[1] + 2)
                    if 0 <= i < cluster_assignments.shape[0] and 0 <= j < cluster_assignments.shape[1]
                ]
                neighboring_clusters.extend(neighbors)

            if neighboring_clusters:
                most_overlapping_cluster = np.argmax(np.bincount(neighboring_clusters))
                refined_cluster_assignments[cluster_pixels] = most_overlapping_cluster

    return refined_cluster_assignments

def segment_image(refined_cluster_assignments):
    segmented_image = np.zeros_like(refined_cluster_assignments, dtype=np.uint8)

    for cluster_index in range(1, np.max(refined_cluster_assignments) + 1):
        cluster_pixels = np.where(refined_cluster_assignments == cluster_index)
        if len(cluster_pixels[0]) > 0:
            segmented_image[cluster_pixels] = 255

    segmented_image = cv2.bitwise_not(segmented_image)

    return segmented_image

def show_results(original, segmented):
    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(122)
    plt.imshow(segmented, cmap='gray', vmin=0, vmax=255)
    plt.title('Resulting Segmented Image')

    plt.tight_layout()
    plt.show()

# Example usage:
image_path = '2nd_year/Intro to AI/midterm/Lisa.jpg'
k = 3
window_size = 3
min_cluster_size = 160

resulting_segmented_image = neutrosophic_kmeans(image_path, k, window_size, min_cluster_size)
original_image = cv2.imread(image_path)
show_results(original_image, resulting_segmented_image)
