# Learner Segmentation and Recommendation Engine

## Research Report

**Project**: EduPro Learner Segmentation
**Goal**: Transition from a generic course suggestion model to a personalized recommendation system to improve learner engagement.

### 1. Exploratory Data Analysis (EDA) and Feature Engineering

The initial dataset consisted of three core entities: `Users`, `Courses`, and `Transactions`.
To build comprehensive learner profiles, we aggregated transactional data to a user level and engineered features across three dimensions:

- **Engagement**: Total courses enrolled, average courses per category, and enrollment frequency (courses per month).
- **Preference**: Category inclinations, level preferences, and average course ratings chosen by the learner.
- **Behavioral**: Average spending per learner, a 'Diversity Score' indicating the breadth of a learner's interests across categories, and a 'Learning Depth Index' mapping the ratio of beginner to advanced courses.

### 2. Segmentation Methodology

We applied **K-Means clustering** to segment the user base.

- **Preprocessing**: Features were standardized using a `StandardScaler` to ensure large variance features (like Spending) did not dominate the distance metrics. Categorical features (Gender, preferred categories) were One-Hot Encoded.
- **Validation**: The optimal number of clusters ($k$) was determined by evaluating the **Silhouette Score**. The results were cross-validated using Agglomerative (Hierarchical) clustering to ensure the stability of the segments.
- **Identified Segments**: Analysis revealed distinct learner archetypes. For example, 'General Explorers' (broad category engagement, lower average spend) versus 'Focused Specialists' (high depth in specific topics, higher spend).

### 3. Recommendation Logic

The recommendation engine leverages a hybrid approach:

1.  **Segment Popularity**: First, we identify the most popular courses within a user's specific cluster based on historical enrollment frequencies.
2.  **Exclusion**: We remove courses the user has already completed.
3.  **Quality Weighting**: Candidate courses are scored based on a weighted combination of their cluster popularity (60%) and their overall course rating (40%).
4.  **Filtering**: The pipeline supports dynamic filtering by Course Level and Course Category.

### 4. Evaluation and Business Impact

- **Intra-Cluster Cohesion**: The segmentation model demonstrated distinct behavioral boundaries between groups, validating the hypothesis that users exhibit non-homogenous learning patterns.
- **Engagement Lift (Proxy)**: By prioritizing courses proven popular among mathematically similar peers (combined with high overall course ratings), the likelihood of a user clicking and enrolling in a recommendation increases significantly over the baseline 'most popular global courses' approach.

### 5. Conclusion

The implementation of this pipeline successfully maps unstructured transaction logs into actionable learner insights. The interactive Streamlit dashboard provides stakeholders with the tools to examine these segments macroscopically and serve personalized content microscopically.
