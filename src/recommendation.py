import pandas as pd
import numpy as np


def load_data_for_recommendation(
    data_dir="z:/Documents/UnifiedMentor/student_segmentation/data",
):
    courses = pd.read_csv(f"{data_dir}/Courses.csv")
    profiles = pd.read_csv(f"{data_dir}/processed/learner_profiles_with_clusters.csv")
    transactions = pd.read_csv(f"{data_dir}/Transactions.csv")
    return courses, profiles, transactions


def get_cluster_popular_courses(cluster_id, profiles, transactions, top_n=10):
    """
    Finds the most popular courses for a given cluster.
    """
    cluster_users = profiles[profiles["Cluster"] == cluster_id]["UserID"]
    cluster_transactions = transactions[transactions["UserID"].isin(cluster_users)]

    popular_courses = cluster_transactions["CourseID"].value_counts().reset_index()
    popular_courses.columns = ["CourseID", "EnrollmentCount"]
    return popular_courses.head(top_n)


def recommend_courses_for_user(
    user_id,
    courses,
    profiles,
    transactions,
    top_n=5,
    filter_level=None,
    filter_category=None,
):
    """
    Recommends courses based on user's cluster popularity and content filtering (ratings).
    """
    if user_id not in profiles["UserID"].values:
        return pd.DataFrame()  # User not found

    user_cluster = profiles[profiles["UserID"] == user_id]["Cluster"].values[0]

    # 1. Get courses popular in this user's cluster
    popular_in_cluster = get_cluster_popular_courses(
        user_cluster, profiles, transactions, top_n=50
    )

    # 2. Filter courses the user has already taken
    user_taken_courses = transactions[transactions["UserID"] == user_id][
        "CourseID"
    ].tolist()
    candidates = popular_in_cluster[
        ~popular_in_cluster["CourseID"].isin(user_taken_courses)
    ]

    # 3. Merge with course details to get ratings, levels, categories
    candidates = candidates.merge(courses, on="CourseID", how="inner")

    # 4. Apply Filters if provided
    if filter_level:
        candidates = candidates[candidates["CourseLevel"] == filter_level]
    if filter_category:
        candidates = candidates[candidates["CourseCategory"] == filter_category]

    # 5. Rank by a combination of Cluster Popularity and Overall Rating
    # We'll normalize both and create a simple engagement score
    if not candidates.empty:
        max_enrolls = candidates["EnrollmentCount"].max()
        if max_enrolls > 0:
            candidates["PopScore"] = candidates["EnrollmentCount"] / max_enrolls
        else:
            candidates["PopScore"] = 0

        candidates["RatingScore"] = candidates["CourseRating"] / 5.0

        # Weighted score: 60% popularity in cluster, 40% overall rating quality
        candidates["RecommendationScore"] = (candidates["PopScore"] * 0.6) + (
            candidates["RatingScore"] * 0.4
        )

        candidates = candidates.sort_values(by="RecommendationScore", ascending=False)

    return candidates.head(top_n)


if __name__ == "__main__":
    courses, profiles, tx = load_data_for_recommendation()

    if not profiles.empty:
        sample_user = profiles["UserID"].iloc[0]
        recs = recommend_courses_for_user(sample_user, courses, profiles, tx)
        print(
            f"Recommendations for user {sample_user} (Cluster {profiles['Cluster'].iloc[0]}):"
        )
        if not recs.empty:
            print(
                recs[
                    [
                        "CourseName",
                        "CourseCategory",
                        "CourseLevel",
                        "RecommendationScore",
                        "CourseRating",
                    ]
                ]
            )
        else:
            print(
                "No recommendations found or user has taken all popular courses in their cluster."
            )
    else:
        print("No user profiles loaded.")
