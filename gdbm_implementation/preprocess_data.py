# kaggle datasets download joebeachcapital/top-10000-spotify-songs-1960-now
# unzip /content/top-10000-spotify-songs-1960-now.zip

import numpy as np
import pandas as pd
import random
import uuid
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv('/content/top_10000_1950-now.csv', encoding='utf-8')
df2 = pd.read_csv('/content/top_10000_1960-now.csv', encoding='utf-8')
df = df1.merge(df2, how='inner', on=["Track Name", "Artist Name(s)", "Album Name"])

print(df.describe(include="all"))

for col in df.columns:
    if col.endswith('_x'):
        base_col = col[:-2]
        if base_col + '_y' in df.columns:
            df[base_col] = df[col].combine_first(df[base_col + '_y'])
            df.drop([col, base_col + '_y'], axis=1, inplace=True)

print(df.columns)

trimmed_df = df[["Track Name", "Artist Name(s)", "Album Release Date", "Track Duration (ms)",
                 "Explicit", "Popularity", "Artist Genres", "Danceability", "Energy", "Key",
                 "Loudness", "Mode", "Speechiness", "Acousticness", "Instrumentalness",
                 "Liveness", "Valence", "Tempo", "Time Signature"]]
trimmed_df = trimmed_df.dropna()

print(trimmed_df.describe())

compiled_genres = trimmed_df["Artist Genres"].str.split(",").explode()
genre_counts = Counter(compiled_genres)
genre_counts_df = pd.DataFrame.from_dict(genre_counts, orient='index', columns=['count'])
genre_counts_df = genre_counts_df.sort_values('count', ascending=False)

print(genre_counts_df.describe())

common_genres = genre_counts_df[genre_counts_df['count'] > 23].index.tolist()
niche_genres = genre_counts_df[genre_counts_df['count'] <= 5].index.tolist()

print(f"Number of common genres: {len(common_genres)}")
print(f"Number of niche genres: {len(niche_genres)}")

num_users = 1000

user_profiles = []

archetypes = {
    'Mainstream Listener': {
        'popularity_range': (50, trimmed_df['Popularity'].max()),
        'energy_range': (0.5, 1.0),
        'danceability_range': (0.5, 1.0),
        'genre_preference': common_genres,  # Prefer common genres
    },
    'Genre-Specific Fan': {
        'popularity_range': (20, trimmed_df['Popularity'].max()),
        'genre_preference': 'genre_specific',  # Will assign a specific common genre
    },
    'Casual Listener': {
        'popularity_range': (30, 70),
        'genre_preference': None,  # Open to all genres
    },
    'Mood-Based Listener': {
        'valence_range': (0.0, 1.0),  # Will vary per user
        'energy_range': (0.0, 1.0),   # Will vary per user
        'genre_preference': None,
    },
    'Loyal Fan': {
        'artist_preference': 'artist_specific',  # Will assign specific artists
    },
    'Discoverer': {
        'popularity_range': (0, 40),
        'genre_preference': None,
    },
    'Niche Genre Enthusiast': {
        'popularity_range': (0, 30),
        'genre_preference': 'niche_genre',  # Will assign a specific niche genre
    },
    'Audiophile': {
        'instrumentalness_range': (0.1, 1.0),
        'acousticness_range': (0.3, 1.0),
        'genre_preference': None,
    },
    'Obscure Music Seeker': {
        'popularity_range': (0, 10),
        'genre_preference': None,
    },
    'Classical Music Lover': {
        'genre_preference': ['classical', 'opera', 'classical tenor'],
        'instrumentalness_range': (0.5, 1.0),
        'danceability_range': (0.0, 0.4),
        'speechiness_range': (0.0, 0.05),
    },
    'Live Music Fan': {
        'liveness_range': (0.3, 1.0),
    },
    'Era-Specific Listener': {
        'release_year_range': (1980, 1989),  # Example for 80s music
    },
}

archetype_distribution = {
    'Mainstream Listener': 0.15,
    'Genre-Specific Fan': 0.15,
    'Casual Listener': 0.15,
    'Mood-Based Listener': 0.10,
    'Loyal Fan': 0.10,
    'Discoverer': 0.10,
    'Niche Genre Enthusiast': 0.05,
    'Audiophile': 0.05,
    'Obscure Music Seeker': 0.05,
    'Classical Music Lover': 0.05,
    'Live Music Fan': 0.05,
    'Era-Specific Listener': 0.05,
}

archetype_track_counts = {
    'Mainstream Listener': 80,
    'Genre-Specific Fan': 70,
    'Casual Listener': 60,
    'Mood-Based Listener': 60,
    'Loyal Fan': 50,
    'Discoverer': 50,
    'Niche Genre Enthusiast': 40,
    'Audiophile': 40,
    'Obscure Music Seeker': 40,
    'Classical Music Lover': 40,
    'Live Music Fan': 40,
    'Era-Specific Listener': 40,
}

for archetype, proportion in archetype_distribution.items():
    num_archetype_users = int(num_users * proportion)
    for _ in range(num_archetype_users):
        profile = {'user_id': uuid.uuid4().hex, 'archetype': archetype}
        archetype_prefs = archetypes[archetype]

        profile.update(archetype_prefs)

        if archetype == 'Mood-Based Listener':
            profile['valence_range'] = tuple(sorted(np.random.uniform(0.0, 1.0, 2)))
            profile['energy_range'] = tuple(sorted(np.random.uniform(0.0, 1.0, 2)))
        if archetype == 'Genre-Specific Fan':
            # Assign a random common genre
            profile['genre_preference'] = [np.random.choice(common_genres)]
        if archetype == 'Niche Genre Enthusiast':
            # Assign a random niche genre
            profile['genre_preference'] = [np.random.choice(niche_genres)]
        if archetype == 'Loyal Fan':
            # Assign a favorite artist
            popular_artists = trimmed_df['Artist Name(s)'].value_counts().head(50).index.tolist()
            profile['artist_preference'] = np.random.choice(popular_artists)
        if archetype == 'Era-Specific Listener':
            # Assign a specific decade randomly
            decades = [(1960, 1969), (1970, 1979), (1980, 1989), (1990, 1999), (2000, 2009), (2010, 2019)]
            profile['release_year_range'] = random.choice(decades)

        user_profiles.append(profile)

user_listening_history = []

for profile in user_profiles:
    user_id = profile['user_id']
    user_tracks = trimmed_df.copy()

    # Apply filters based on user preferences
    if 'popularity_range' in profile:
        user_tracks = user_tracks[
            (user_tracks['Popularity'] >= profile['popularity_range'][0]) &
            (user_tracks['Popularity'] <= profile['popularity_range'][1])
        ]
    if 'energy_range' in profile:
        user_tracks = user_tracks[
            (user_tracks['Energy'] >= profile['energy_range'][0]) &
            (user_tracks['Energy'] <= profile['energy_range'][1])
        ]
    if 'danceability_range' in profile:
        user_tracks = user_tracks[
            (user_tracks['Danceability'] >= profile['danceability_range'][0]) &
            (user_tracks['Danceability'] <= profile['danceability_range'][1])
        ]
    if 'valence_range' in profile:
        user_tracks = user_tracks[
            (user_tracks['Valence'] >= profile['valence_range'][0]) &
            (user_tracks['Valence'] <= profile['valence_range'][1])
        ]
    if 'acousticness_range' in profile:
        user_tracks = user_tracks[
            (user_tracks['Acousticness'] >= profile['acousticness_range'][0]) &
            (user_tracks['Acousticness'] <= profile['acousticness_range'][1])
        ]
    if 'instrumentalness_range' in profile:
        user_tracks = user_tracks[
            (user_tracks['Instrumentalness'] >= profile['instrumentalness_range'][0]) &
            (user_tracks['Instrumentalness'] <= profile['instrumentalness_range'][1])
        ]
    if 'speechiness_range' in profile:
        user_tracks = user_tracks[
            (user_tracks['Speechiness'] >= profile['speechiness_range'][0]) &
            (user_tracks['Speechiness'] <= profile['speechiness_range'][1])
        ]
    if 'liveness_range' in profile:
        user_tracks = user_tracks[
            (user_tracks['Liveness'] >= profile['liveness_range'][0]) &
            (user_tracks['Liveness'] <= profile['liveness_range'][1])
        ]
    if 'genre_preference' in profile and profile['genre_preference']:
        preferred_genres = profile['genre_preference']
        user_tracks = user_tracks[
            user_tracks['Artist Genres'].apply(lambda genres: any(genre in genres for genre in preferred_genres))
        ]
    if 'artist_preference' in profile and profile['artist_preference']:
        user_tracks = user_tracks[
            user_tracks['Artist Name(s)'] == profile['artist_preference']
        ]
    if 'release_year_range' in profile:
        user_tracks['Album Release Date'] = pd.to_datetime(user_tracks['Album Release Date'], errors='coerce')
        user_tracks = user_tracks[user_tracks['Album Release Date'].notnull()].copy()
        user_tracks['Release Year'] = user_tracks['Album Release Date'].dt.year
        user_tracks = user_tracks[
            (user_tracks['Release Year'] >= profile['release_year_range'][0]) &
            (user_tracks['Release Year'] <= profile['release_year_range'][1])
        ]

    if user_tracks.empty:
        user_tracks = trimmed_df.sample(n=50)

    num_tracks_listened = archetype_track_counts[profile['archetype']]

    if len(user_tracks) >= num_tracks_listened:
        listened_tracks = user_tracks.sample(n=num_tracks_listened, replace=False)
    else:
        listened_tracks = user_tracks.sample(n=num_tracks_listened, replace=True)

    for _, track in listened_tracks.iterrows():
        interaction = {
            'user_id': user_id,
            'track_name': track['Track Name'],
            'artist_name': track['Artist Name(s)'],
            'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365)),
            'archetype': profile['archetype']
        }
        user_listening_history.append(interaction)

for profile in user_profiles:
    if np.random.rand() < 0.1:
        user_id = profile['user_id']
        random_tracks = trimmed_df.sample(n=5)
        for _, track in random_tracks.iterrows():
            interaction = {
                'user_id': user_id,
                'track_name': track['Track Name'],
                'artist_name': track['Artist Name(s)'],
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365)),
                'archetype': profile['archetype'],
            }
            user_listening_history.append(interaction)


def assign_rating(track_row, profile):
  rating = 3
  if 'genre_preference' in profile and profile['genre_preference']:
    if any(genre in track_row['Artist Genres'] for genre in profile['genre_preference']):
      rating += 1
  if 'artist_preference' in profile and profile['artist_preference']:
    if track_row['Artist Name(s)'] == profile['artist_preference']:
      rating += 1
  if 'popularity_range' in profile:
    mid_popularity = np.mean(profile['popularity_range'])
    if track_row['Popularity'] >= mid_popularity:
      rating += 0.5
  if 'instrumentalness_range' in profile:
    mid_instrumentalness = np.mean(profile['instrumentalness_range'])
    if track_row['Instrumentalness'] >= mid_instrumentalness:
      rating += 0.5
  if 'liveness_range' in profile:
    mid_liveness = np.mean(profile['liveness_range'])
    if track_row['Liveness'] >= mid_liveness:
      rating += 0.5
  return min(max(rating, 1), 5)

interaction_df = pd.DataFrame(user_listening_history)
interaction_df = interaction_df.merge(trimmed_df, left_on='track_name', right_on='Track Name', how='left')
interaction_df['rating'] = interaction_df.apply(
    lambda x: assign_rating(x, next(profile for profile in user_profiles if profile['user_id'] == x['user_id'])),
    axis=1
)
print(interaction_df.describe())
print(interaction_df.columns)
archetype_counts = interaction_df['archetype'].value_counts()
print("Archetype Distribution in Interaction Data:")
print(archetype_counts)

# ****************************************************************************************************************
# EDA
# ****************************************************************************************************************

num_users = interaction_df['user_id'].nunique()
print(f"Total unique users: {num_users}")


num_tracks = interaction_df['track_name'].nunique()
print(f"Total unique tracks: {num_tracks}")

archetype_counts = interaction_df.groupby('archetype')['user_id'].nunique().reset_index(name='num_users')
interaction_counts = interaction_df['archetype'].value_counts().reset_index(name='num_interactions')
archetype_summary = pd.merge(archetype_counts, interaction_counts, left_on='archetype', right_on='archetype')
print(archetype_summary)

plt.figure(figsize=(8, 6))
sns.histplot(interaction_df['rating'], bins=5, kde=False)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

average_rating = interaction_df['rating'].mean()
print(f"Average Rating: {average_rating:.2f}")

plt.figure(figsize=(12, 6))
sns.boxplot(x='archetype', y='rating', data=interaction_df)
plt.xticks(rotation=45)
plt.title('Ratings by User Archetype')
plt.xlabel('Archetype')
plt.ylabel('Rating')
plt.show()

user_interactions = interaction_df.groupby('user_id').size().reset_index(name='interactions')
plt.figure(figsize=(8, 6))
sns.histplot(user_interactions['interactions'], bins=30, kde=False)
plt.title('Distribution of Interactions per User')
plt.xlabel('Number of Interactions')
plt.ylabel('Count of Users')
plt.show()

avg_interactions_per_user = user_interactions['interactions'].mean()
print(f"Average Interactions per User: {avg_interactions_per_user:.2f}")

interactions_per_archetype = interaction_df.groupby('archetype')['user_id'].count() / interaction_df.groupby('archetype')['user_id'].nunique()
interactions_per_archetype = interactions_per_archetype.reset_index(name='avg_interactions')
plt.figure(figsize=(12, 6))
sns.barplot(x='archetype', y='avg_interactions', data=interactions_per_archetype)
plt.xticks(rotation=45)
plt.title('Average Interactions per User by Archetype')
plt.xlabel('Archetype')
plt.ylabel('Average Number of Interactions')
plt.show()

audio_features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Instrumentalness', 'Liveness', 'Speechiness']

for feature in audio_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(interaction_df[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()

plt.figure(figsize=(10, 8))
corr = interaction_df[audio_features].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Audio Features')
plt.show()

for feature in audio_features:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='archetype', y=feature, data=interaction_df)
    plt.xticks(rotation=45)
    plt.title(f'{feature} by User Archetype')
    plt.xlabel('Archetype')
    plt.ylabel(feature)
    plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='archetype', y='Popularity', data=interaction_df)
plt.xticks(rotation=45)
plt.title('Track Popularity by User Archetype')
plt.xlabel('Archetype')
plt.ylabel('Popularity')
plt.show()

interaction_df['timestamp'] = pd.to_datetime(interaction_df['timestamp'])
interaction_df.set_index('timestamp', inplace=True)
interactions_over_time = interaction_df.resample('M').size()

plt.figure(figsize=(12, 6))
interactions_over_time.plot()
plt.title('Total Interactions Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Interactions')
plt.show()

interactions_by_archetype = interaction_df.groupby(['timestamp', 'archetype']).size().unstack().resample('M').sum()
interactions_by_archetype.plot.area(figsize=(12, 6), stacked=True)
plt.title('Interactions Over Time by Archetype')
plt.xlabel('Month')
plt.ylabel('Number of Interactions')
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()

interaction_df_copy = interaction_df.copy(deep=True)

interaction_df_copy = interaction_df_copy[interaction_df_copy['Artist Genres'].notna()]
interaction_df_copy['Artist Genres'] = interaction_df_copy['Artist Genres'].str.split(',')
interaction_df_copy['Artist Genres'] = interaction_df_copy['Artist Genres'].apply(
    lambda x: [genre.strip() for genre in x]
)
interaction_df_exploded = interaction_df_copy.explode('Artist Genres')
interaction_df_exploded = interaction_df_exploded[interaction_df_exploded['Artist Genres'] != '']
top_genres = interaction_df_exploded['Artist Genres'].value_counts().head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_genres.values, y=top_genres.index)
plt.title('Top 20 Genres in the Dataset')
plt.xlabel('Number of Interactions')
plt.ylabel('Genre')
plt.show()

top_genres_list = top_genres.index.tolist()
genre_archetype_matrix = interaction_df_exploded[interaction_df_exploded['Artist Genres'].isin(top_genres_list)]
genre_archetype_counts = genre_archetype_matrix.groupby(['archetype', 'Artist Genres']).size().unstack(fill_value=0)

plt.figure(figsize=(12, 8))
sns.heatmap(genre_archetype_counts, annot=False, cmap='Blues')
plt.title('Genre Preferences by Archetype')
plt.xlabel('Genre')
plt.ylabel('Archetype')
plt.show()

interaction_df.to_csv('interaction_data.csv', index=False)
user_profiles_df = pd.DataFrame(user_profiles)
user_profiles_df.to_csv('user_profiles.csv', index=False)