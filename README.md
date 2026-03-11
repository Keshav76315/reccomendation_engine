# EduPro Learner Segmentation & Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)

## Overview

This repository contains an end-to-end Machine Learning pipeline and personalized recommendation dashboard built for **EduPro**, an e-learning platform.

> **Note:** This is a mini-project developed during my Data Science Internship at [Unified Mentor](https://www.unifiedmentor.com/). It demonstrates the practical application of clustering algorithms to solve real-world user engagement challenges in the EdTech sector.

## Project Goal

EduPro's existing system serves generic course suggestions to all learners. This project replaces that one-size-fits-all approach by:

1.  **Segmenting** learners based on their demographic, preference, and behavioral data.
2.  **Recommending** courses tailored to a user's specific segment (e.g., matching a "General Explorer" with broad, highly-rated introductory courses while matching a "Specialist" with deep-dive technical content).

## Features

- **Data Processing Pipeline**: Aggregates user transactions to compute Engagement, Preference, and Behavioral scores (e.g., total spend, category diversity).
- **K-Means Clustering**: Unsupervised machine learning model grouping users into distinct, mathematically validated segments.
- **Hybrid Recommendation Logic**: Personalized course suggestions blending cluster popularity with overall course rating quality.
- **Interactive Dashboard**: A Streamlit application allowing stakeholders to explore learner profiles, visualize cluster distributions (PCA, Radar charts), and test the recommendation engine in real-time.

## Project Structure

```text
.
├── app.py                      # Main Streamlit Dashboard Application
├── data/                       # Raw datasets (Users, Courses, Transactions, etc.)
│   └── processed/              # Engineered profiles & cluster features
├── reports/                    # Final project deliverables
│   ├── executive_summary.md    # High-level business impact overview
│   └── research_paper.md       # Technical methodology and EDA findings
├── src/                        # Core ML Pipeline Modules
│   ├── data_preprocessing.py   # Data merging and feature engineering logic
│   ├── modeling.py             # K-Means clustering and validation metrics
│   └── recommendation.py       # Personalized course recommendation logic
├── requirements.txt            # Project dependencies
└── temp/                       # Temporary workspace for ad-hoc analysis scripts
```

## Quickstart

**1. Setup Environment**
Ensure you have Python 3.10+ installed. Create and activate a virtual environment:

```bash
python -m venv .venv
# On Windows
.\.venv\Scripts\Activate.ps1
# On MacOS/Linux
source .venv/bin/activate
```

**2. Install Dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the ML Pipeline (Optional)**
_Note: The processed data and models are already generated, but you can re-run the pipeline with these commands:_

```bash
python src/data_preprocessing.py
python src/modeling.py
```

**4. Launch the Dashboard**

```bash
streamlit run app.py
```

## Tools & Libraries Used

- **pandas & numpy**: Data manipulation and numerical operations.
- **scikit-learn**: K-Means clustering, StandardScaler, Silhouette metrics.
- **Streamlit**: Rapid web application prototyping.
- **Plotly**: Interactive data visualizations.
