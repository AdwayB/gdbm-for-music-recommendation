import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import expit

def activation_function(x):
  # return np.tanh(x)
  # return 1 / (1 + np.exp(-x))
  return expit(x)

class RBM:
  def __init__(self, n_visible, n_hidden):
    self.n_visible = n_visible
    self.n_hidden = n_hidden
    self.log_sigma_squared = np.zeros(n_visible)
    limit = np.sqrt(6 / (n_visible + n_hidden))
    self.weights = np.random.uniform(-limit, limit, size=(n_visible, n_hidden))
    self.visible_bias = np.zeros(n_visible)
    self.hidden_bias = np.zeros(n_hidden)

  def sample_hidden(self, visible):
    sigma_squared = np.exp(self.log_sigma_squared)
    activation = (np.dot(visible / sigma_squared, self.weights) + self.hidden_bias)
    probabilities = activation_function(activation)
    hidden_sample = np.random.binomial(1, probabilities)
    return probabilities, hidden_sample

  def sample_visible(self, hidden):
    sigma_squared = np.exp(self.log_sigma_squared)
    mean = np.dot(hidden, self.weights.T) + self.visible_bias
    visible_sample = mean + np.random.normal(0, np.sqrt(sigma_squared), size=mean.shape)
    return visible_sample

  def train(self, data, learning_rate=0.001, epochs=10, batch_size=100, weight_decay=0.0001):
    lr_weights = learning_rate
    lr_biases = learning_rate
    lr_variance = learning_rate * 0.5
    max_grad = 5.0

    num_examples = data.shape[0]
    for epoch in range(epochs):
      np.random.shuffle(data)
      batch_errors = []
      for i in range(0, num_examples, batch_size):
        batch = data[i:i + batch_size]

        sigma_squared = np.exp(self.log_sigma_squared)

        pos_hidden_probs, pos_hidden_states = self.sample_hidden(batch)
        pos_associations = np.dot((batch / sigma_squared).T, pos_hidden_probs)

        neg_visible = self.sample_visible(pos_hidden_states)
        neg_hidden_probs, _ = self.sample_hidden(neg_visible)
        neg_associations = np.dot((neg_visible / sigma_squared).T, neg_hidden_probs)

        grad_w = (pos_associations - neg_associations) / batch_size

        grad_w -= weight_decay * self.weights

        grad_vb = np.mean((batch - neg_visible) / sigma_squared, axis=0)
        grad_hb = np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)
        delta_log_sigma_squared = (np.mean(((batch - self.visible_bias) ** 2) / sigma_squared - 1, axis=0) -
                                   np.mean(((neg_visible - self.visible_bias) ** 2) / sigma_squared - 1, axis=0))

        grad_w = np.clip(grad_w, -max_grad, max_grad)
        grad_vb = np.clip(grad_vb, -max_grad, max_grad)
        grad_hb = np.clip(grad_hb, -max_grad, max_grad)
        delta_log_sigma_squared = np.clip(delta_log_sigma_squared, -max_grad, max_grad)

        self.weights += lr_weights * grad_w
        self.visible_bias += lr_biases * grad_vb
        self.hidden_bias += lr_biases * grad_hb
        self.log_sigma_squared += lr_variance * delta_log_sigma_squared

        min_log_sigma = -4
        max_log_sigma = 4
        self.log_sigma_squared = np.clip(self.log_sigma_squared, min_log_sigma, max_log_sigma)

        if not np.all(np.isfinite(self.weights)):
          print("Weights have non-finite values.")
        if not np.all(np.isfinite(self.visible_bias)):
          print("Visible biases have non-finite values.")
        if not np.all(np.isfinite(self.hidden_bias)):
          print("Hidden biases have non-finite values.")

        error = np.mean((batch - neg_visible) ** 2)
        if np.isnan(error) or np.isinf(error):
          print("Reconstruction error is NaN or Inf.")
        batch_errors.append(error)

      epoch_error = np.mean(batch_errors)
      print(f"Epoch {epoch + 1}/{epochs}, Reconstruction Error: {epoch_error:.4f}")


class DBM:
  def __init__(self, layer_sizes):
    self.rbms = []
    for i in range(len(layer_sizes) - 1):
      rbm = RBM(layer_sizes[i], layer_sizes[i + 1])
      self.rbms.append(rbm)

  def pretrain_layers(self, data, learning_rate=0.01, epochs=10, batch_size=100, weight_decay=0.0001):
    input_data = data
    for idx, rbm in enumerate(self.rbms):
      print(f"Pre-training RBM Layer {idx + 1}/{len(self.rbms)}")
      rbm.train(input_data, learning_rate, epochs, batch_size, weight_decay)
      input_data, _ = rbm.sample_hidden(input_data)

  def finetune(self, data, learning_rate=0.01, epochs=10, batch_size=100):
    num_examples = data.shape[0]
    for epoch in range(epochs):
      np.random.shuffle(data)
      batch_errors = []
      for i in range(0, num_examples, batch_size):
        batch = data[i:i + batch_size]

        activations = [batch]
        v = batch
        for rbm in self.rbms:
          h_mean, h_sample = rbm.sample_hidden(v)
          activations.append(h_mean)
          v = h_mean

        v = activations[-1]
        for idx in reversed(range(len(self.rbms))):
          rbm = self.rbms[idx]
          v = rbm.sample_visible(v)
          h_mean_neg, h_sample_neg = rbm.sample_hidden(v)

          v_pos = activations[idx]
          h_pos = activations[idx + 1]

          sigma_squared = np.exp(rbm.log_sigma_squared)

          pos_associations = np.dot((v_pos / sigma_squared).T, h_pos)
          neg_associations = np.dot((v / sigma_squared).T, h_mean_neg)

          rbm.weights += learning_rate * (pos_associations - neg_associations) / batch_size
          rbm.visible_bias += learning_rate * np.mean((v_pos - v) / sigma_squared, axis=0)
          rbm.hidden_bias += learning_rate * np.mean(h_pos - h_mean_neg, axis=0)

          # Update log variance
          pos_var_term = ((v_pos - rbm.visible_bias) ** 2) / sigma_squared
          neg_var_term = ((v - rbm.visible_bias) ** 2) / sigma_squared
          delta_log_sigma_squared = learning_rate * (np.mean(pos_var_term - 1, axis=0) - np.mean(neg_var_term - 1, axis=0))
          rbm.log_sigma_squared += delta_log_sigma_squared

          min_log_sigma = -4  # Corresponds to variance ≈ 0.018
          max_log_sigma = 4  # Corresponds to variance ≈ 54.6
          rbm.log_sigma_squared = np.clip(rbm.log_sigma_squared, min_log_sigma, max_log_sigma)

          v = v_pos

        error = np.mean((batch - v) ** 2)
        batch_errors.append(error)

      print(f"Fine-tuning Epoch {epoch + 1}/{epochs}")

  def forward_pass(self, data):
    input_data = data
    for rbm in self.rbms:
      input_data = rbm.sample_hidden(input_data)[0]
    return input_data

  def reconstruct(self, data):
    activations = [data]
    v = data
    for rbm in self.rbms:
      h_mean, h_sample = rbm.sample_hidden(v)
      activations.append(h_mean)
      v = h_mean

    for idx in reversed(range(len(self.rbms))):
      rbm = self.rbms[idx]
      v = rbm.sample_visible(v)

    return v


df = pd.read_csv('../interaction_data.csv')

df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)

interaction_df = df[['user_id', 'track_name']].drop_duplicates()
interaction_df['interaction'] = 1

user_song_matrix = interaction_df.pivot(index='user_id', columns='track_name', values='interaction').fillna(0)

song_feature_cols = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode',
                     'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness',
                     'Valence', 'Tempo', 'Time Signature', 'Popularity', 'Track Duration (ms)']
song_features = df[['track_name'] + song_feature_cols].drop_duplicates(subset='track_name').set_index('track_name')
song_features_normalized = (song_features - song_features.mean()) / song_features.std()
song_features_matrix = song_features_normalized.values

num_visible = song_features_matrix.shape[1]
num_epochs = 30
learningRate = 0.001
weightDecay = 0.0005

dbm = DBM([num_visible, 1024, 512, 256])
dbm.pretrain_layers(song_features_matrix, learning_rate=learningRate, epochs=num_epochs, batch_size=128, weight_decay=weightDecay)
dbm.finetune(song_features_matrix, learning_rate=learningRate, epochs=num_epochs, batch_size=128)

song_embeddings = dbm.reconstruct(song_features_matrix)
song_embeddings_df = pd.DataFrame(song_embeddings, index=song_features_normalized.index)

user_listened_songs = interaction_df.groupby('user_id')['track_name'].apply(list)
user_preference_vectors = {}
for userId, songs_list in user_listened_songs.items():
    embeddings = song_embeddings_df.loc[songs_list]
    user_preference_vectors[userId] = embeddings.mean(axis=0)

user_preferences_df = pd.DataFrame(user_preference_vectors).T

def recommend_songs(user_id, user_preferences, songEmbeddings, userSongMatrix, top_n=10):
  user_vector = user_preferences.loc[user_id].values.reshape(1, -1)
  similarities = cosine_similarity(user_vector, songEmbeddings.values)[0]
  similarity_series = pd.Series(similarities, index=songEmbeddings.index)
  listened_songs = userSongMatrix.columns[userSongMatrix.loc[user_id] > 0]
  recommendations = similarity_series.drop(listened_songs, errors='ignore')
  top_recommendations = recommendations.sort_values(ascending=False).head(top_n)
  return top_recommendations

user_to_recommend = "2edc5b296b0948cabc6dd754d0250e43"
recommended_songs = recommend_songs(user_to_recommend, user_preferences_df, song_embeddings_df, user_song_matrix, top_n=10)

print(f"Top recommendations for {user_to_recommend}:\n{recommended_songs}")
