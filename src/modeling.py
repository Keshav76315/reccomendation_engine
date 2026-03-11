import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import os
import joblib


def load_processed_data(
    data_dir="z:/Documents/UnifiedMentor/student_segmentation/data/processed",
):
    features = pd.read_csv(f"{data_dir}/clustering_features.csv")
    full_profiles = pd.read_csv(f"{data_dir}/learner_profiles.csv")
    return features, full_profiles


def find_optimal_clusters(features, max_k=10):
    """
    Evaluates KMeans from k=2 to max_k using Silhouette Score.
    """
    best_k = 2
    best_score = -1
    models = {}

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        models[k] = kmeans

        print(f"k={k}, Silhouette Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"\nOptimum number of clusters found: {best_k} with score {best_score:.4f}")
    return best_k, models[best_k]


def validate_with_hierarchical(features, n_clusters):
    """
    Runs Agglomerative Clustering as a validation step.
    """
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hc.fit_predict(features)
    score = silhouette_score(features, labels)
    print(f"Hierarchical Clustering (k={n_clusters}) Silhouette Score: {score:.4f}")
    return labels


def train_and_save_model(
    features,
    full_profiles,
    data_dir="z:/Documents/UnifiedMentor/student_segmentation/data/processed",
):
    print("Finding optimal k...")
    best_k, best_model = find_optimal_clusters(features)

    print("\nValidating with Hierarchical Clustering...")
    validate_with_hierarchical(features, best_k)

    # Assign cluster labels to the profiles
    cluster_labels = best_model.predict(features)
    features["Cluster"] = cluster_labels
    full_profiles["Cluster"] = cluster_labels

    # Map cluster numbers to semantic names based on characteristics (we'll generate simplistic names here and refine in analysis later if needed)
    # 0 -> Explorers, 1 -> Specialists, 2 -> Career-Focused (example mapping, robust mapping is done in the app via analysis)
    # We will just save the raw cluster IDs for now

    print("\nSaving model and updated profiles...")
    os.makedirs(f"{data_dir}/models", exist_ok=True)
    joblib.dump(best_model, f"{data_dir}/models/kmeans_model.pkl")
    full_profiles.to_csv(f"{data_dir}/learner_profiles_with_clusters.csv", index=False)
    features.to_csv(f"{data_dir}/clustering_features_with_clusters.csv", index=False)

    print("Modeling completed.")


if __name__ == "__main__":
    features, profiles = load_processed_data()
    train_and_save_model(features, profiles)
