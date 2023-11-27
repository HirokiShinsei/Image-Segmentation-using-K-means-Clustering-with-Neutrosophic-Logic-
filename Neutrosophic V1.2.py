import cv2
import numpy as np
import matplotlib.pyplot as plt

def grayscale_conversion(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def neutrosophic_falsity(mean_image):
    falsity_image = 1 - mean_image
    return falsity_image

def neutrosophic_indeterminacy(mean_image, std_image, diff_image):
    # indeterminacy_image = (std_image * diff_image) / np.maximum(std_image, diff_image)
    # return indeterminacy_image
    
    indeterminacy_image = np.where((std_image == 0) & (diff_image == 0),0,(std_image * diff_image) / np.maximum(std_image, diff_image))

    return indeterminacy_image

def kmeans_clustering(data, k, max_iterations=10):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(max_iterations):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=-1)
        labels = np.argmin(distances, axis=-1)

        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)

    return labels

def neutrosophic_clustering(truth_image, falsity_image, indeterminacy_image, k):
    neutrosophic_image = np.stack([truth_image, falsity_image, indeterminacy_image], axis=-1)
    neutrosophic_image_reshaped = neutrosophic_image.reshape((-1, 3))

    cluster_assignments = kmeans_clustering(neutrosophic_image_reshaped, k)

    cluster_assignments = cluster_assignments.reshape(truth_image.shape)

    return cluster_assignments

def refine_clusters(cluster_assignments, min_cluster_size):
    refined_cluster_assignments = np.copy(cluster_assignments)

    for cluster_index in range(np.max(cluster_assignments) + 1):
        cluster_pixels = np.where(cluster_assignments == cluster_index)

        if len(cluster_pixels[0]) < min_cluster_size:
            neighboring_clusters = []
            for pixel in zip(*cluster_pixels):
                neighbors = [(pixel[0] + i, pixel[1] + j) for i in range(-1, 2) for j in range(-1, 2)]
                neighbors = [(x, y) for x, y in neighbors if 0 <= x < cluster_assignments.shape[0] and 0 <= y < cluster_assignments.shape[1] and cluster_assignments[x, y] != cluster_index]
                neighboring_clusters.extend(cluster_assignments[x, y] for x, y in neighbors)

            most_overlapping_cluster = np.argmax(np.bincount(neighboring_clusters))
            refined_cluster_assignments[cluster_pixels] = most_overlapping_cluster

    return refined_cluster_assignments

def segment_image(refined_cluster_assignments):
    segmented_image = np.zeros_like(refined_cluster_assignments, dtype=np.uint8)

    # cluster_labels = {}
    # for cluster_index in range(np.max(refined_cluster_assignments) + 1):
    #     cluster_label = len(cluster_labels)
    #     cluster_labels[cluster_index] = cluster_label
    #     segmented_image[refined_cluster_assignments == cluster_index] = cluster_label

    # Assign a unique intensity to each cluster.
    # unique_clusters = np.unique(refined_cluster_assignments)
    # for i, cluster_index in enumerate(unique_clusters):
    #     segmented_image[refined_cluster_assignments == cluster_index] = int(255 * i / (len(unique_clusters) - 1))
    
    # segmented_image[refined_cluster_assignments > 0] = 255
    
    segmented_image = np.where(refined_cluster_assignments > 0, 0, 255).astype(np.uint8)

    return segmented_image

if __name__ == "__main__":
    
    image_path = "2nd_year/Intro to AI/midterm/Lisa.jpg"
    k = 2
    min_cluster_size = 160

    gray_image = grayscale_conversion(image_path)
    mean_image = gray_image / 255.0

    falsity_image = neutrosophic_falsity(mean_image)

    image = cv2.imread(image_path)
    std_image = np.std(image, axis=-1) / 255.0
    diff_image = np.abs(mean_image - std_image)
    indeterminacy_image = neutrosophic_indeterminacy(mean_image, std_image, diff_image)

    cluster_assignments = neutrosophic_clustering(mean_image, falsity_image, indeterminacy_image, k)
    print("Cluster Assignments:", cluster_assignments)
    refined_cluster_assignments = refine_clusters(cluster_assignments, min_cluster_size)
    print("Refined Cluster Assignments:", refined_cluster_assignments)
    segmented_image = segment_image(refined_cluster_assignments)

    # Debugging prints
    print("Mean Image:", mean_image)
    print("Falsity Image:", falsity_image)
    print("Indeterminacy Image:", indeterminacy_image)
    
    # Inside the main script
    plt.imshow(mean_image, cmap="gray")
    plt.title("Mean Image")
    plt.show()

    plt.imshow(falsity_image, cmap="gray")
    plt.title("Falsity Image")
    plt.show()

    plt.imshow(cluster_assignments, cmap="viridis")
    plt.title("Cluster Assignments")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image, cmap="gray")
    plt.title("Segmented Image")

    plt.show()
