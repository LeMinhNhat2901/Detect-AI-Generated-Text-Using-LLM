# 🎵 Spotify Music Trends & Audio Fingerprinting EDA

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0-blue.svg)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

> "Music is a mirror of society." In the digital age, data is the magnifying glass that lets us see that mirror with stunning clarity. Why do we dance with a broken heart? How has TikTok reshaped the very structure of a hit song?

This project embarks on an analytical journey into the vast universe of Spotify data, exploring over **135,000 songs** to decode the trends shaping today's music industry. From creating unique "Audio Fingerprints" for genres to explaining modern musical paradoxes like "Sad Bops," this EDA uncovers the stories hidden within the soundwaves.

---

## 📚 Table of Contents

1.  [Project Overview & Team Info 👥](#1-project-overview--team-info-)
2.  [Dataset Description 📊](#2-dataset-description-)
3.  [Key Research Questions ❓](#3-key-research-questions-)
4.  [Key Findings & Insights 💡](#4-key-findings--insights-)
5.  [File Structure 📂](#5-file-structure-)
6.  [How to Run the Project 🚀](#6-how-to-run-the-project-)
7.  [Technologies & Dependencies 📦](#7-technologies--dependencies-)
8.  [Contact & Contribution 📧](#8-contact--contribution-)
9.  [Academic Information 🎓](#9-academic-information-)

---

## 1. Project Overview & Team Info 👥

### Overview

This project focuses on Exploratory Data Analysis (EDA) of a large-scale Spotify dataset. The primary goal is to leverage statistical analysis and advanced data visualization to answer meaningful questions about modern music. Additionally, the project includes the development of a content-based recommendation system that suggests songs based on their acoustic features.

### Team Members

| No. | Student Name | Student ID | Role |
| :-- | :--- | :--- | :--- |
| 1 | **Lê Minh Nhật** | 23120067 | Project Lead, EDA, Meaningful Questions, Recommended app demo |
| 2 | **Huỳnh Đặng Ngọc Hân** | 23120042 | Data Preprocessing, Modeling, Data Collection |
| 3 | **Nguyễn Duy Khánh** | 23120051 | Modeling, Documentation |

---

## 2. Dataset Description 📊

*   **Data Source:** The dataset was collected via the Spotify Web API, containing over 114,000 tracks.
*   **Primary Data File:** `data/dataset.csv`
*   **Key Features (20+ acoustic attributes):**
    *   **`popularity`**: A score from 0-100 indicating the song's popularity.
    *   **`duration_ms`**: The duration of the track in milliseconds.
    *   **`explicit`**: A boolean indicating if the track has explicit content.
    *   **`acousticness`**: A confidence measure from 0.0 to 1.0 of whether the track is acoustic.
    *   **`danceability`**: Describes how suitable a track is for dancing based on a combination of musical elements.
    *   **`energy`**: A perceptual measure of intensity and activity.
    *   **`instrumentalness`**: Predicts whether a track contains no vocals.
    *   **`liveness`**: Detects the presence of an audience in the recording.
    *   **`loudness`**: The overall loudness of a track in decibels (dB).
    *   **`speechiness`**: Detects the presence of spoken words in a track.
    *   **`tempo`**: The overall estimated tempo of a track in beats per minute (BPM).
    *   **`valence`**: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track.

---

## 3. Key Research Questions ❓

This analysis seeks to answer several compelling questions about the music landscape:

1.  **Audio Fingerprinting:** What are the unique sonic signatures of top music genres (e.g., Pop, Rock, Hip-hop, EDM)? Can we visualize them using Radar Charts?
2.  **Modern Music Paradoxes:**
    *   **'Sad Bops':** Why do we love dancing to sad songs? An analysis of the relationship between `Energy` and `Valence`.
    *   **'The Loudness War':** Is music actually getting louder? A look at the correlation between `Acousticness` and `Loudness` over time.
3.  **The TikTok Era:** How has the rise of TikTok since 2018 influenced song duration and acoustic characteristics?
4.  **Explicit Content & Genre:** Is there a statistically significant relationship between explicit lyrics and music genre? (Analyzed using Chi-square test and Cramer's V).

---

## 4. Key Findings & Insights 💡

Our analysis revealed several fascinating trends:

*   **Distinct Genre Fingerprints:** Radar charts show that genres have highly distinct audio profiles. For instance, **Acoustic/Folk** is characterized by high `acousticness` and low `energy`, whereas **EDM/Dance** exhibits the opposite pattern. This confirms that acoustic features are powerful differentiators.

*   **The "Sad Bop" Phenomenon is Real:** A significant cluster of popular songs features high `energy` (> 0.6) but low `valence` (< 0.4). This data-driven insight confirms the trend of upbeat tracks with melancholic themes.

*   **The TikTok Effect is Measurable:** Songs released post-2018 show a **statistically significant decrease in duration** and a noticeable **increase in `danceability`** compared to the pre-TikTok era, suggesting a structural shift towards shorter, more engaging tracks.

*   **Hip-hop/Rap's Strong Link to Explicit Content:** The Chi-square test yielded a very low p-value, indicating a strong statistical association between the **Hip-hop/Rap** genre and tracks labeled as `explicit`.

### Unique Visualizations
This project utilizes unique visualization techniques to tell a clearer story:
*   **3x2 Radar Chart Layouts:** To compare the "Audio Fingerprints" of multiple genres simultaneously.
*   **Annotated Heatmaps:** To display the results of the Chi-square test and correlations between features effectively.

---

## 5. File Structure 📂

```text
spotify_songs_recommendation/
├── .gitignore                      # Specifies intentionally untracked files to ignore
│
├── app/                            # Streamlit Web Application
│   ├── app.py                      # Main Streamlit application
│   ├── train_model.py              # Script to train/export ML model
│   ├── requirements_app.txt        # Dependencies for the app
│   ├── README_APP.md               # Detailed guide for running the app
│   └── audio_cache/                # Cached audio files (auto-generated)
│
├── data/                           # Contains all data files
│   ├── dataset.csv                 # The primary raw dataset
│   ├── final_dataset_ml.csv        # Final dataset for machine learning
│   ├── final_features.npz          # Final features for machine learning
│   ├── preprocessed_dataset.csv    # Preprocessed dataset
│   ├── spotify_playlist.json       # Spotify playlist data
│   ├── step1_metadata.csv          # Metadata from step 1
│   ├── test_dataset_bonus.csv      # Test dataset for bonus part
│   ├── theoretical_ground_truth.json # Theoretical ground truth data
│   ├── train_dataset_bonus.csv     # Train dataset for bonus part
│   ├── training_feature_extraction_dataset.csv # Dataset for training feature extraction
│   └── used_seeds.json             # Used seeds for data collection
│
├── notebooks/                      # Jupyter Notebooks for analysis
│   ├── Create_playlist.ipynb       # Notebook to create a playlist
│   ├── Data_collection.ipynb       # Scripts for data gathering
│   ├── Data_exploring.ipynb        # General exploratory data analysis
│   ├── Data_modeling_bonus.ipynb   # Model building for bonus part
│   ├── Data_preprocessing.ipynb    # Data cleaning and preprocessing steps
│   ├── Data_preprocessing_bonus.ipynb # Data preprocessing for bonus part
│   ├── Meaningful_questions.ipynb  # Notebook dedicated to answering key questions
│   ├── Recommend_system.ipynb      # Implementation of the recommendation system
│   └── Run_model_feature.ipynb     # Notebook to run model and extract features
│
├── src/                            # Source code for helper modules
│   ├── audio_downloader.py         # Script to download audio
│   ├── audio_extractor.py          # Script to extract audio features
│   ├── extractor_model.py          # Model for feature extraction
│   ├── find_playlist.py            # Script to find playlists
│   ├── scraping_from_api.py        # Script to scrape data from API
│   └── models/                     # Saved models or preprocessing objects
│       ├── preprocessing_models.joblib # Saved preprocessing models
│       └── spotify_model.pkl       # Trained ML model for recommendations
│
├── README.md                       # This file
├── requirements.txt                # Required libraries for the project
└── project_summary.pdf             # Project summary document
```

---

## 6. How to Run the Project 🚀

### Prerequisites
*   Python 3.8 or higher
*   `pip` and `venv` (or `conda`)

### 6.1: Clone the Repository
```bash
git clone https://github.com/LeMinhNhat2901/spotify_songs_recommendation.git
cd spotify_songs_recommendation
```

### 6.2: Create and Activate a Virtual Environment (Recommended)
Using `venv`:
```bash
# Create the environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```
Or using `conda`:
```bash
conda create -n spotify-eda python=3.9
conda activate spotify-eda
```

### 6.3: Install Dependencies
```bash
pip install -r requirements.txt
```

### 6.4: Run the Notebooks
Launch Jupyter Notebook and open the files in the `notebooks/` directory. It is recommended to follow the logical order, starting with data collection and preprocessing before moving to exploration and modeling.

### 6.5: Run the Streamlit Web Application 🎵

The project includes a beautiful web application for music recommendations!

```bash
# Install app dependencies
pip install -r app/requirements_app.txt

# Run the Streamlit app
streamlit run app/app.py
```

Then open your browser at `http://localhost:8501`

**Features of the Web App:**
- 🎧 **Listen to music directly** via Spotify embed player
- 📜 **Build listening history** by selecting 5-10 songs you like
- 🎯 **Get personalized recommendations** based on your taste
- 💾 **Export playlists** as CSV or open in Spotify
- 🧹 **Auto cache cleanup** for downloaded audio

👉 **For detailed instructions, see [app/README_APP.md](app/README_APP.md)**

---

## 7. Technologies & Dependencies 📦

*   **Data Manipulation:** `pandas`, `numpy`
*   **Data Visualization:** `matplotlib`, `seaborn`, `plotly`
*   **Statistical Analysis:** `scipy`
*   **Machine Learning:** `scikit-learn`
*   **Development Environment:** `Jupyter Notebook`

A complete list is available in `requirements.txt`.

---

## 8. Contact & Contribution 📧

Questions, feedback, and contributions are welcome! Please feel free to open an issue or contact the project lead.

*   **Project Lead:** Lê Minh Nhật - [GitHub Profile](https://github.com/LeMinhNhat2901)
*   **Email:** nhat29012005@gmail.com

---

## 9. Academic Information 🎓

### Course Details

*   **Course Name:** Introduction to Data Science
*   **Department:** Faculty of Information Technology, University of Science (VNU-HCM)
*   **Academic Year:** 2025-2026

### Instructors

*   **Lê Ngọc Thành** - <lnthanh@fit.hcmus.edu.vn>
*   **Lê Nhựt Nam** - <lnnam@fit.hcmus.edu.vn>
*   **Huỳnh Lâm Hải Đăng** - <hlhdang@fit.hcmus.edu.vn>
*   **Võ Nam Thục Đoan** - <vntdoan@fit.hcmus.edu.vn>

### Academic Integrity Commitment

This project adheres to the highest standards of academic integrity:

*   All source code is original work or properly cited.
*   References, libraries, and external resources are clearly acknowledged.
*   Collaboration is limited to permitted group discussions and teamwork as allowed by the course.
*   No plagiarism or unauthorized code sharing has occurred.

### Support and Questions

For course-related questions or technical support:

*   **Primary Channel:** Use the course's ZALO group.
*   **Office Hours:** As announced by the instructors.
*   **Project Issues:** Contact the project lead via email or GitHub.

---

**© 2025 University of Science (VNU-HCMC)**  
*Developed for Programming for data science*

---

**⭐ If you find this project helpful, please consider giving it a star!**
