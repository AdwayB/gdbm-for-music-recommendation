# gdbm-for-music-recommendation

---

## Overview
This project is a music recommendation system that combines a Deep Boltzmann Machine (DBM) with a sophisticated preprocessing pipeline to simulate user listening behavior and generate personalized song suggestions. The DBM learns hierarchical latent representations of audio features, while the preprocessing script creates realistic user interaction data based on archetype-based user modeling.

The recommendation system is designed to:
- Learn rich embeddings from song audio features.
- Simulate user behavior using predefined archetypes.
- Recommend songs based on user preferences derived from their listening history.

---

## **Preprocessing Pipeline**

The preprocessing pipeline transforms raw datasets into a structured, rich format suitable for training and evaluation. Two datasets, containing Spotify song metadata and audio features, are merged and cleaned to produce a unified dataset.

### **Key Steps**

1. **Data Cleaning and Integration**:
   - The datasets are merged on common attributes (`Track Name`, `Artist Name(s)`, and `Album Name`), resolving duplicate columns and filling missing values.
   - A trimmed DataFrame is created with relevant attributes such as:
     - **Audio Features**: Danceability, Energy, Loudness, Acousticness, etc.
     - **Metadata**: Popularity, Explicit flag, Artist Genres, and Album Release Date.

2. **Genre Analysis**:
   - Artist genres are exploded into individual genres.
   - Common genres (frequent in the dataset) and niche genres (less frequent) are identified, providing a foundation for genre-specific recommendations.

3. **User Archetype Simulation**:
   - **12 Archetypes** are defined to represent diverse listening behaviors, such as:
     - **Mainstream Listener**: Prefers popular and energetic tracks.
     - **Mood-Based Listener**: Focuses on valence and energy ranges.
     - **Niche Genre Enthusiast**: Enjoys rare or unconventional genres.
     - **Era-Specific Listener**: Listens to music from a specific decade.
   - Each archetype has specific preferences (e.g., popularity, genre, energy) which influence the simulated listening behavior.

4. **Listening Histories**:
   - Users (1,000 simulated) are assigned tracks based on their archetype preferences.
   - Each interaction is timestamped and associated with a rating computed based on how well the track aligns with the user’s preferences.
   - Additional random interactions are added to simulate discovery and exploration.

5. **Output Data**:
   - **`interaction_data.csv`**: Contains user-track interactions, ratings, and archetype metadata.
   - **`user_profiles.csv`**: Stores simulated user profiles, including archetype definitions and preferences.

---

## **Recommendation System**

The recommendation system uses a Gaussian-Bernoulli Deep Boltzmann Machine with Bernoulli hidden units and Gaussian visible units to generate personalized song suggestions. The GDBM learns latent embeddings from audio features, which are then used to model user preferences.

### **Key Components**

1. **Data Preparation**:
   - The interaction data is converted into a **User-Song Matrix** (binary interactions) and a **Song Feature Matrix** for training.

2. **Gaussian-Bernoulli Deep Boltzmann Machine (DBM)**:
   - A stack of Gaussian-Bernoulli Restricted Boltzmann Machines (GRBMs) with progressively smaller layers.
   - Trained in two phases:
     - **Pretraining**: Each layer learns independently, capturing hierarchical feature representations, which facilitates better performance during fine-tuning.
     - **Fine-Tuning**: The entire DBM is trained end-to-end to minimize reconstruction error.

3. **Mean-Field Inference**:
   - A parallel mean-field inference algorithm is used to approximate the posterior distribution of the hidden units given the visible data.
   - The hidden states are used to sample the visible layer, which is then used to reconstruct the song embeddings.

4. **Block Gibbs Sampling**:
   - A parallel block Gibbs sampler is used to generate negative samples from the model distribution, which are used in the contrastive divergence step during fine-tuning.

5. **Fine-Tuning Algorithm**:
   - The contrastive divergence step computes gradients by contrasting positive associations from the data with negative associations from model-generated samples.
   - Regularization methods such as weight decay and gradient clipping are applied to prevent exploding values, ensuring stable parameter updates.
   - Uses the Adam optimization algorithm to adaptively adjust learning rates for each parameter, enhancing convergence efficiency and training stability.

6. **User Preferences**:
   - Each user’s preference vector is computed as the mean embedding of the songs they interacted with.

7. **Recommendation Generation**:
   - Cosine similarity is calculated between a user’s preference vector and all song embeddings.
   - Songs already listened to by the user are excluded.
   - The top `N` most similar songs are returned as recommendations.

--- 

### **Preprocessing**

1. **Download the Datasets**:
   ```bash
   kaggle datasets download joebeachcapital/top-10000-spotify-songs-1960-now
   unzip top-10000-spotify-songs-1960-now.zip