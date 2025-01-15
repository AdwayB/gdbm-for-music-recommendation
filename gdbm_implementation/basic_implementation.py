import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import expit

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

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
  """
      A multi-layer DBM. Each 'RBM' in self.rbms connects layer i (bottom) to layer i+1 (top).
      For example, if layer_sizes = [nV, nH1, nH2], then:
        self.rbms[0] connects V <-> H1
        self.rbms[1] connects H1 <-> H2
      Implements:
        - Full mean-field inference (up–down) with top-down influences
        - Full block Gibbs sampling (down–up) with each layer being sampled
      """
  def __init__(self, layer_sizes):
    self.rbms = []
    for i in range(len(layer_sizes) - 1):
      rbm = RBM(layer_sizes[i], layer_sizes[i + 1])
      self.rbms.append(rbm)

    self.adam_buffers = []
    for rbm in self.rbms:
      buffer_dict = {
        'v_w': np.zeros_like(rbm.weights),
        's_w': np.zeros_like(rbm.weights),
        'v_vb': np.zeros_like(rbm.visible_bias),
        's_vb': np.zeros_like(rbm.visible_bias),
        'v_hb': np.zeros_like(rbm.hidden_bias),
        's_hb': np.zeros_like(rbm.hidden_bias),
        'v_log_sigma': np.zeros_like(rbm.log_sigma_squared),
        's_log_sigma': np.zeros_like(rbm.log_sigma_squared),
      }
      self.adam_buffers.append(buffer_dict)

  def pretrain_layers(self, data, learning_rate=0.01, epochs=10, batch_size=100, weight_decay=0.0001):
    input_data = data
    for idx, rbm in enumerate(self.rbms):
      print(f"Pre-training RBM Layer {idx + 1}/{len(self.rbms)}")
      rbm.train(input_data, learning_rate, epochs, batch_size, weight_decay)
      input_data, _ = rbm.sample_hidden(input_data)

  def mean_field_inference(self, visible_data, num_iters=5):
    """
    A parallel mean-field inference for the entire DBM.
     - At each iteration, every hidden layer i is updated in parallel
       using the previous iteration's states for neighbors.

    Returns a list of arrays: [h1_mean, h2_mean, ..., hN_mean].
    """
    n_layers = len(self.rbms)
    batch_size = visible_data.shape[0]

    hidden_means = [0.5 * np.ones((batch_size, rbm.n_hidden))
                    for rbm in self.rbms]

    # For Gaussian bottom RBM
    bottom_rbm = self.rbms[0]
    sigma_squared = np.exp(bottom_rbm.log_sigma_squared)

    for _ in range(num_iters):
      old_means = [hm.copy() for hm in hidden_means]

      for i, rbm in enumerate(self.rbms):
        # Bottom-up contribution
        if i == 0:
          # This hidden layer sees the visible data (Gaussian) plus top-down from layer i+1
          pre_act = np.dot(visible_data / sigma_squared, rbm.weights) + rbm.hidden_bias
        else:
          # Middle or top hidden sees hidden_means[i-1]
          pre_act = np.dot(old_means[i - 1], rbm.weights) + rbm.hidden_bias

        # Top-down contribution
        if i < n_layers - 1:
          top_rbm = self.rbms[i + 1]
          pre_act += np.dot(old_means[i + 1], top_rbm.weights.T)

        hidden_means[i] = activation_function(pre_act)

    return hidden_means

  def block_gibbs_sampling(self, hidden_init_list, k=5):
    """
    A parallel block Gibbs sampler for the negative phase in a DBM.
     - In each sub-step:
        1) Re-sample all hidden layers *in parallel* (each sees neighbors from old state).
        2) Re-sample the visible from the bottom hidden.

    hidden_init_list: initial hidden states [h1, h2, ...] (e.g., from mean field).
    Returns (v_neg, hidden_states), where
      v_neg is the final visible sample,
      hidden_states = [h1_neg, h2_neg, ...].
    """
    n_layers = len(self.rbms)
    hidden_states = [h.copy() for h in hidden_init_list]

    bottom_rbm = self.rbms[0]
    sigma_squared = np.exp(bottom_rbm.log_sigma_squared)
    v_neg = bottom_rbm.sample_visible(hidden_states[0])

    for _ in range(k):
      old_hidden = [hs.copy() for hs in hidden_states]

      # Re-sample each hidden layer in parallel
      for i, rbm in enumerate(self.rbms):
        # bottom-up contribution
        if i == 0:
          pre_act = np.dot(v_neg / sigma_squared, rbm.weights) + rbm.hidden_bias
        else:
          pre_act = np.dot(old_hidden[i - 1], rbm.weights) + rbm.hidden_bias

        # top-down contribution
        if i < n_layers - 1:
          top_rbm = self.rbms[i + 1]
          pre_act += np.dot(old_hidden[i + 1], top_rbm.weights.T)

        # Stochastic sampling of hidden units
        probs = activation_function(pre_act)
        hidden_states[i] = (np.random.rand(*probs.shape) < probs).astype(float)

      # Re-sample visible from the bottom hidden
      v_neg = bottom_rbm.sample_visible(hidden_states[0])

    return v_neg, hidden_states

  def finetune(self, data, learning_rate=0.001, epochs=10, batch_size=100,
               mean_field_iters=5, gibbs_k=5, max_grad=3.0, weight_decay=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Full DBM fine-tuning with:
      1) Mean-field inference (with up–down sweeps) => positive phase
      2) Block Gibbs sampling (with down–up passes) => negative phase
      3) Parameter updates using difference of associations, plus optional weight decay
    """
    num_examples = data.shape[0]
    t = 0

    for epoch in range(epochs):
      np.random.shuffle(data)
      batch_errors = []

      for start_idx in range(0, num_examples, batch_size):
        t += 1
        batch = data[start_idx:start_idx + batch_size]
        bsz = batch.shape[0]

        # Positive Phase
        hidden_means = self.mean_field_inference(batch, num_iters=mean_field_iters)

        # Negative Phase
        v_neg, hidden_states_neg = self.block_gibbs_sampling(hidden_means, k=gibbs_k)

        for i, rbm in enumerate(self.rbms):
          sigma_squared = np.exp(rbm.log_sigma_squared)

          if i == 0:
            pos_assoc = np.dot((batch / sigma_squared).T, hidden_means[0]) / bsz
            neg_assoc = np.dot((v_neg / sigma_squared).T, hidden_states_neg[0]) / bsz
            grad_w = pos_assoc - neg_assoc - weight_decay * rbm.weights
            grad_vb = (
                np.mean(batch / sigma_squared, axis=0)
                - np.mean(v_neg / sigma_squared, axis=0)
            )
            grad_hb = (
                np.mean(hidden_means[0], axis=0)
                - np.mean(hidden_states_neg[0], axis=0)
            )
            delta_log_sigma = (
                np.mean(((batch - rbm.visible_bias) ** 2) / sigma_squared - 1, axis=0)
                - np.mean(((v_neg - rbm.visible_bias) ** 2) / sigma_squared - 1, axis=0)
            )
          else:
            lower_h_pos = hidden_means[i - 1]
            upper_h_pos = hidden_means[i]
            lower_h_neg = hidden_states_neg[i - 1]
            upper_h_neg = hidden_states_neg[i]
            pos_assoc = (np.dot((lower_h_pos / sigma_squared).T, upper_h_pos) / bsz)
            neg_assoc = (np.dot((lower_h_neg / sigma_squared).T, upper_h_neg) / bsz)
            grad_w = pos_assoc - neg_assoc - weight_decay * rbm.weights
            grad_vb = (
                np.mean(lower_h_pos / sigma_squared, axis=0)
                - np.mean(lower_h_neg / sigma_squared, axis=0)
            )
            grad_hb = (
                np.mean(upper_h_pos, axis=0)
                - np.mean(upper_h_neg, axis=0)
            )
            delta_log_sigma = np.zeros_like(rbm.log_sigma_squared)

          grad_w = np.clip(grad_w, -max_grad, max_grad)
          grad_vb = np.clip(grad_vb, -max_grad, max_grad)
          grad_hb = np.clip(grad_hb, -max_grad, max_grad)
          delta_log_sigma = np.clip(delta_log_sigma, -max_grad, max_grad)

          self.adam_buffers[i]['v_w'] = beta1 * self.adam_buffers[i]['v_w'] + (1 - beta1) * grad_w
          self.adam_buffers[i]['v_vb'] = beta1 * self.adam_buffers[i]['v_vb'] + (1 - beta1) * grad_vb
          self.adam_buffers[i]['v_hb'] = beta1 * self.adam_buffers[i]['v_hb'] + (1 - beta1) * grad_hb
          self.adam_buffers[i]['v_log_sigma'] = beta1 * self.adam_buffers[i]['v_log_sigma'] + (
                1 - beta1) * delta_log_sigma

          self.adam_buffers[i]['s_w'] = beta2 * self.adam_buffers[i]['s_w'] + (1 - beta2) * (grad_w ** 2)
          self.adam_buffers[i]['s_vb'] = beta2 * self.adam_buffers[i]['s_vb'] + (1 - beta2) * (grad_vb ** 2)
          self.adam_buffers[i]['s_hb'] = beta2 * self.adam_buffers[i]['s_hb'] + (1 - beta2) * (grad_hb ** 2)
          self.adam_buffers[i]['s_log_sigma'] = beta2 * self.adam_buffers[i]['s_log_sigma'] + (1 - beta2) * (
                delta_log_sigma ** 2)

          m_w_hat = self.adam_buffers[i]['v_w'] / (1 - beta1 ** t)
          v_w_hat = self.adam_buffers[i]['s_w'] / (1 - beta2 ** t)
          m_vb_hat = self.adam_buffers[i]['v_vb'] / (1 - beta1 ** t)
          v_vb_hat = self.adam_buffers[i]['s_vb'] / (1 - beta2 ** t)
          m_hb_hat = self.adam_buffers[i]['v_hb'] / (1 - beta1 ** t)
          v_hb_hat = self.adam_buffers[i]['s_hb'] / (1 - beta2 ** t)
          m_log_hat = self.adam_buffers[i]['v_log_sigma'] / (1 - beta1 ** t)
          v_log_hat = self.adam_buffers[i]['s_log_sigma'] / (1 - beta2 ** t)

          rbm.weights += -learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
          rbm.visible_bias += -learning_rate * m_vb_hat / (np.sqrt(v_vb_hat) + epsilon)
          rbm.hidden_bias += -learning_rate * m_hb_hat / (np.sqrt(v_hb_hat) + epsilon)
          rbm.log_sigma_squared += -learning_rate * 0.5 * m_log_hat / (np.sqrt(v_log_hat) + epsilon)

        mse = np.mean((batch - v_neg) ** 2)
        batch_errors.append(mse)

      print(f"Fine-Tuning Epoch {epoch + 1}/{epochs}, MSE: {np.mean(batch_errors):.4f}")

  def forward_pass(self, data):
    input_data = data
    for rbm in self.rbms:
      input_data = rbm.sample_hidden(input_data)[0]
    return input_data

  def reconstruct(self, data):
    """
    Reconstruct data by going up (prob_h) then down (sample_visible)
    through each layer in reverse.
    """
    # Up
    hidden_samples = []
    current = data
    for rbm in self.rbms:
      _, h_samp = rbm.sample_hidden(current)
      hidden_samples.append(h_samp)
      current = h_samp

    # Down
    for i in reversed(range(len(self.rbms))):
      rbm = self.rbms[i]
      current = rbm.sample_visible(hidden_samples[i])
    return current


df = pd.read_csv('../interaction_data.csv')

df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)

interaction_df = df[['user_id', 'track_name']].drop_duplicates()
interaction_df['interaction'] = 1

artist_map = (
  df[['track_name', 'artist_name']]
  .drop_duplicates(subset='track_name')
  .set_index('track_name')['artist_name']
  .to_dict()
)

user_song_matrix = interaction_df.pivot(index='user_id', columns='track_name', values='interaction').fillna(0)

song_feature_cols = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode',
                     'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness',
                     'Valence', 'Tempo', 'Time Signature', 'Popularity', 'Track Duration (ms)']
song_features = df[['track_name'] + song_feature_cols].drop_duplicates(subset='track_name').set_index('track_name')
song_features_normalized = (song_features - song_features.mean()) / song_features.std()
song_features_matrix = song_features_normalized.values

num_visible = song_features_matrix.shape[1]
num_epochs = 100
learningRate = 0.005
weightDecay = 0.0005

dbm = DBM([num_visible, 16, 32, 64, 32, 16])

dbm.pretrain_layers(song_features_matrix,
                    learning_rate=learningRate,
                    epochs=num_epochs,
                    batch_size=128,
                    weight_decay=weightDecay)

dbm.finetune(song_features_matrix,
             learning_rate=0.0001,
             epochs=50,
             batch_size=128,
             mean_field_iters=3,
             gibbs_k=3,
             max_grad=3.0,
             weight_decay=0.001,
             beta1=0.9,
             beta2=0.9)

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

  recommendation_df = top_recommendations.reset_index()
  recommendation_df.columns = ['track_name', 'similarity_score']
  recommendation_df['artist_name'] = recommendation_df['track_name'].map(artist_map)
  recommendation_df = recommendation_df[['track_name', 'artist_name', 'similarity_score']]
  return recommendation_df

user_to_recommend = "2edc5b296b0948cabc6dd754d0250e43"
recommended_songs = recommend_songs(user_to_recommend, user_preferences_df, song_embeddings_df, user_song_matrix, top_n=20)

print(f"Top recommendations for {user_to_recommend}:\n{recommended_songs}")
