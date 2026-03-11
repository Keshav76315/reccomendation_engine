import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os


def load_data(data_dir="z:/Documents/UnifiedMentor/student_segmentation/data"):
    users = pd.read_csv(f"{data_dir}/Users.csv")
    courses = pd.read_csv(f"{data_dir}/Courses.csv")
    transactions = pd.read_csv(f"{data_dir}/Transactions.csv")
    return users, courses, transactions


def create_learner_profiles(users, courses, transactions):
    # Merge transactions with Course details
    tx_courses = transactions.merge(courses, on="CourseID", how="left")

    # Engagement Features
    engagement = (
        tx_courses.groupby("UserID")
        .agg(
            TotalCoursesEnrolled=("CourseID", "count"),
            UniqueCategoriesEnrolled=("CourseCategory", "nunique"),
        )
        .reset_index()
    )
    engagement["AverageCoursesPerCategory"] = engagement[
        "TotalCoursesEnrolled"
    ] / engagement["UniqueCategoriesEnrolled"].replace(0, 1)

    # Enrollment Frequency
    transactions["TransactionDate"] = pd.to_datetime(
        transactions["TransactionDate"], format="%d/%m/%Y"
    )
    dates = (
        transactions.groupby("UserID")
        .agg(FirstTx=("TransactionDate", "min"), LastTx=("TransactionDate", "max"))
        .reset_index()
    )
    dates["DaysActive"] = (dates["LastTx"] - dates["FirstTx"]).dt.days + 1
    # Frequency: Enrolls per 30 days
    freq = engagement[["UserID", "TotalCoursesEnrolled"]].merge(dates, on="UserID")
    freq["EnrollmentFrequency"] = (
        freq["TotalCoursesEnrolled"] / freq["DaysActive"]
    ) * 30
    engagement = engagement.merge(
        freq[["UserID", "EnrollmentFrequency"]], on="UserID", how="left"
    )

    # Transactional Features
    spending = (
        transactions.groupby("UserID")
        .agg(TotalSpending=("Amount", "sum"), AverageSpending=("Amount", "mean"))
        .reset_index()
    )

    # Preference Features
    category_prefs = (
        tx_courses.groupby(["UserID", "CourseCategory"]).size().unstack(fill_value=0)
    )
    category_prefs = category_prefs.add_prefix("Cat_").reset_index()

    level_prefs = (
        tx_courses.groupby(["UserID", "CourseLevel"]).size().unstack(fill_value=0)
    )
    level_prefs = level_prefs.add_prefix("Level_").reset_index()

    avg_rating = (
        tx_courses.groupby("UserID")
        .agg(AverageCourseRating=("CourseRating", "mean"))
        .reset_index()
    )

    # Combine all engineered features
    profiles = users.merge(engagement, on="UserID", how="left")
    profiles = profiles.merge(spending, on="UserID", how="left")
    profiles = profiles.merge(category_prefs, on="UserID", how="left")
    profiles = profiles.merge(level_prefs, on="UserID", how="left")
    profiles = profiles.merge(avg_rating, on="UserID", how="left")

    # Fill any NaNs resulting from merges (though every user should have at least one transaction in this dataset based on our previous EDA)
    profiles.fillna(0, inplace=True)

    # Additional Behavioral Feature: Diversity Score (ratio of unique categories to total courses)
    profiles["DiversityScore"] = profiles["UniqueCategoriesEnrolled"] / profiles[
        "TotalCoursesEnrolled"
    ].replace(
        0, 1
    )  # Avoid div by zero

    # Learning Depth Index (beginner vs advanced ratio)
    # If Level_Advanced is 0, we can add 1 to denominator to prevent infinite ratio
    if "Level_Beginner" in profiles.columns and "Level_Advanced" in profiles.columns:
        profiles["LearningDepthIndex"] = profiles["Level_Beginner"] / (
            profiles["Level_Advanced"] + 1
        )
    else:
        profiles["LearningDepthIndex"] = 0.0

    return profiles


def preprocess_for_clustering(profiles):
    features_to_scale = [
        "Age",
        "TotalCoursesEnrolled",
        "UniqueCategoriesEnrolled",
        "AverageCoursesPerCategory",
        "EnrollmentFrequency",
        "TotalSpending",
        "AverageSpending",
        "AverageCourseRating",
        "DiversityScore",
        "LearningDepthIndex",
    ]
    # Add category and level columns dynamically based on what was unstacked
    cat_cols = [c for c in profiles.columns if c.startswith("Cat_")]
    lvl_cols = [c for c in profiles.columns if c.startswith("Level_")]
    features_to_scale.extend(cat_cols)
    features_to_scale.extend(lvl_cols)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(profiles[features_to_scale])

    scaled_df = pd.DataFrame(
        scaled_data, columns=features_to_scale, index=profiles.index
    )

    # One-Hot Encoding Gender
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    gender_encoded = encoder.fit_transform(profiles[["Gender"]])
    gender_df = pd.DataFrame(
        gender_encoded,
        columns=encoder.get_feature_names_out(["Gender"]),
        index=profiles.index,
    )

    # Final matrix for K-Means: UserID is kept out
    final_features = pd.concat([scaled_df, gender_df], axis=1)

    return final_features, profiles, scaler, encoder


if __name__ == "__main__":
    users, courses, transactions = load_data()
    profiles = create_learner_profiles(users, courses, transactions)
    features, full_profiles, scaler, encoder = preprocess_for_clustering(profiles)

    print("Preprocessing completed.")
    print(f"Final feature matrix shape: {features.shape}")
    print(f"Sample full profile columns: {full_profiles.columns.tolist()[:10]}...")

    # Create processed dir if it doesn't exist to save outputs for easy access in app and notebooks
    os.makedirs(
        "z:/Documents/UnifiedMentor/student_segmentation/data/processed", exist_ok=True
    )
    full_profiles.to_csv(
        "z:/Documents/UnifiedMentor/student_segmentation/data/processed/learner_profiles.csv",
        index=False,
    )
    features.to_csv(
        "z:/Documents/UnifiedMentor/student_segmentation/data/processed/clustering_features.csv",
        index=False,
    )
