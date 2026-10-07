✨ RecomAI — AI-Powered Beauty Product Recommendation Engine

RecomAI is a machine learning-based beauty product recommendation system built using Amazon Beauty product interaction data.

The system analyzes historical user-product interactions and compares multiple recommendation approaches to generate personalized product recommendations.

It also provides an interactive Streamlit interface where users can select a recommendation model, explore products, and view personalized recommendations.

---

 🚀 Project Overview

Online stores contain thousands of products, making it difficult for users to discover products relevant to their interests.

RecomAI addresses this problem using collaborative filtering and machine learning techniques.

The project implements and compares:

- Singular Value Decomposition (SVD)
- Item-Based Collaborative Filtering
- K-Nearest Neighbors (KNN)
- Popularity-Based Recommendation
- Hybrid Recommendation

The final system provides an interactive web interface built with Streamlit.

---

 ✨ Features

 🤖 Multiple Recommendation Models

RecomAI supports four main recommendation approaches:

| Model | Description |
|---|---|
| **SVD** | Learns latent user and product factors to predict ratings |
| **Item-CF** | Recommends products similar to products the user interacted with |
| **KNN** | Uses nearest-neighbor search with cosine similarity |
| **Hybrid** | Combines SVD, Item-CF and popularity signals |

### 🔎 Product Explorer

Users can:

- Search beauty products
- Filter products by rating
- View product information
- View product images
- Find similar products

### 📊 Model Analytics

The application provides:

- Dataset statistics
- Precision@5
- Recall@5
- NDCG@5
- MAE
- RMSE
- Model comparison

### 🎨 Interactive UI

The application is built using Streamlit with a custom dark UI using the RecomAI branding.

---

# 🧠 Recommendation Architecture

```text
                    Amazon Beauty Dataset
                            │
                            ▼
                    Data Preprocessing
                            │
                            ▼
                  User-Product Interactions
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
            SVD          Item-CF          KNN
             │              │              │
             └──────────────┼──────────────┘
                            │
                            ▼
                    Recommendation Layer
                            │
                    ┌───────┴────────┐
                    │                │
                    ▼                ▼
               Popularity        Hybrid Model
                    │                │
                    └───────┬────────┘
                            │
                            ▼
                    Top-N Recommendations
                            │
                            ▼
                     Streamlit Interface
