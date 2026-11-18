# PathSnap – GPS Map Matching and Route Reconstruction

## Project Description

PathSnap is a machine learning-based GPS map-matching and route reconstruction system. The project aims to identify the correct road segments from noisy GPS observations and reconstruct a geographically plausible vehicle route.

The system combines machine learning algorithms with a Hidden Markov Model (HMM) and Viterbi decoding to improve route matching by considering both GPS observations and road network connectivity.

## Team Members – Team 5

| Name | SRN |
|---|---|
| Ishanika Vinuthan | PES2UG24CS911 |
| Ayush Kumar | PES2UG24CS101 |
| Bhavani R Shahapur | PES2UG24CS114 |
| Bheema Varshini | PES2UG24CS115 |

## Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Hyperparameter Optimization:** Optuna (Bayesian Optimization)
- **Route Reconstruction:** Hidden Markov Model and Viterbi Algorithm
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Visualization:** Folium
- **Version Control:** Git and GitHub

## Dataset

The project uses GPS trajectory data and road network information.

- Training dataset: 80 trips
- Student-test dataset: 10 trips
- GPS observations stored in `.track` files
- Road network nodes stored in `.nodes` files
- Road segments stored in `.arcs` files
- Reference routes stored in `.route` files

## Methodology

### 1. Data Preprocessing

- Load GPS trajectories and road network data.
- Clean and validate GPS observations.
- Generate the 10 nearest candidate road segments.
- Extract distance, heading, speed, time-gap, displacement and candidate-rank features.
- Align reference routes to generate training labels.

### 2. Machine Learning Models

The following algorithms were evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Histogram-based Gradient Boosting

### 3. Bayesian Optimization

Optuna's Tree-structured Parzen Estimator (TPE) was used to optimize Random Forest hyperparameters.

The tuned parameters include:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `class_weight`

The best recorded cross-validation Macro F1-score was **83.30%**.

### 4. HMM and Viterbi Algorithm

The Hidden Markov Model represents candidate road segments as hidden states.

- Emission scores represent how well GPS observations match candidate roads.
- Transition scores consider road connectivity and vehicle movement.
- Viterbi decoding identifies a likely sequence of road segments.

### 5. Web Application

The application uses:

- **FastAPI** to expose backend endpoints and connect the route reconstruction pipeline to the frontend.
- **Streamlit** to provide an interactive interface.
- **Folium** to visualize GPS trajectories and reconstructed routes.

The architecture supports running the frontend and backend on separate laptops connected through the same network.

## Model Comparison

| Model | Validation Top-1 Accuracy |
|---|---:|
| Logistic Regression | 76.95% |
| KNN | 73.08% |
| Decision Tree | 75.51% |
| Random Forest | 76.68% |
| HistGradientBoosting | 76.68% |
| Bayesian-optimized Random Forest | 75.57% |
| HMM + Viterbi | 58.87% |

*These results represent the recorded validation experiment. The Bayesian-optimized Random Forest's 83.30% cross-validation Macro F1-score is a different evaluation metric.*

## Project Workflow

1. Load GPS and road network datasets.
2. Preprocess GPS observations.
3. Generate candidate road segments.
4. Extract candidate features.
5. Train and compare machine learning models.
6. Optimize Random Forest using Bayesian Optimization.
7. Apply HMM and Viterbi for route reconstruction.
8. Evaluate predicted routes.
9. Visualize results using Streamlit and Folium.

## Repository

[PathSnap – GPS Map Matching](https://github.com/ishanikavinuthan/pathsnap-map-matching)

## Academic Information

**University:** PES University  
**Department:** Computer Science and Engineering  
**Team:** 5

## Conclusion

PathSnap explores how machine learning and probabilistic sequence modeling can be combined for GPS map matching. The project evaluates candidate-level road classification and route-level reconstruction, while also investigating the effect of road connectivity on prediction accuracy.
