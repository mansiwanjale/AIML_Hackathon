import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

# ==============================================================================
# CLUSTER NAMES MAPPING
# ==============================================================================

CLUSTER_NAMES = {
    0: "Mental Health & Neurodiversity",
    1: "Psychology & Happiness",
    2: "Design & Innovation",
    3: "Photography & Visual Arts",
    4: "Entrepreneurship & Business",
    5: "Biology & Evolution",
    6: "Healthcare & Medicine",
    7: "Physics & Space",
    8: "Poetry & Performance Art",
    9: "Gender & Equality",
    10: "Climate & Environment",
    11: "Philosophy & Human Nature",
    12: "Cancer & Medical Research",
    13: "Global Issues & Politics",
    14: "Social Change & Activism",
    15: "Data & Digital Technology",
    16: "Robotics & Future Tech",
    17: "Film & Entertainment",
    18: "Education & Culture",
    19: "Art & Creativity",
    20: "Music & Live Performance",
    21: "Energy & Sustainability",
    22: "Ocean & Marine Science",
    23: "Neuroscience & Cognition",
    24: "Architecture & Urban Design"
}

CLUSTER_CATEGORIES = {
    0: "Health & Science",
    1: "Social Sciences",
    2: "Technology & Design",
    3: "Arts & Media",
    4: "Business & Innovation",
    5: "Life Sciences",
    6: "Health & Science",
    7: "Physical Sciences",
    8: "Arts & Media",
    9: "Social Sciences",
    10: "Environment & Sustainability",
    11: "Humanities",
    12: "Health & Science",
    13: "Global Affairs",
    14: "Social Sciences",
    15: "Technology & Design",
    16: "Technology & Design",
    17: "Arts & Media",
    18: "Education & Culture",
    19: "Arts & Media",
    20: "Arts & Media",
    21: "Environment & Sustainability",
    22: "Environment & Sustainability",
    23: "Health & Science",
    24: "Technology & Design"
}

CLUSTER_DESCRIPTIONS = {
    0: "Speakers focusing on mental health awareness, autism spectrum, depression, and psychological wellbeing",
    1: "Experts in positive psychology, happiness research, workplace satisfaction, and behavioral economics",
    2: "Innovators in product design, user interfaces, creative technology, and design thinking",
    3: "Visual storytellers, photographers, and artists working with imagery and visual media",
    4: "Entrepreneurs, startup founders, and business innovators driving change through enterprise",
    5: "Biologists studying DNA, evolution, animal behavior, and biodiversity conservation",
    6: "Healthcare professionals, public health experts, and medical practitioners advancing patient care",
    7: "Physicists, astronomers, and cosmologists exploring the universe, space, and fundamental laws",
    8: "Poets, spoken word artists, and performance artists expressing through verse and voice",
    9: "Advocates for gender equality, women's rights, feminism, and social justice",
    10: "Climate scientists, environmentalists, and sustainability experts addressing planetary challenges",
    11: "Philosophers, linguists, and thinkers exploring human nature, compassion, and meaning",
    12: "Medical researchers and scientists pioneering cancer treatment and disease prevention",
    13: "Global affairs experts, policy makers, and activists addressing international challenges",
    14: "Social activists and community organizers driving societal transformation and impact",
    15: "Data scientists, internet pioneers, and digital privacy advocates shaping the online world",
    16: "Roboticists, automation engineers, and future technology developers",
    17: "Filmmakers, game designers, and entertainment industry professionals",
    18: "Educators, learning experts, and cultural commentators reshaping how we teach and learn",
    19: "Visual artists, creative professionals, and aesthetic innovators",
    20: "Musicians, composers, and live performers creating sonic experiences",
    21: "Renewable energy experts, clean tech innovators, and sustainable power advocates",
    22: "Marine biologists, ocean conservationists, and underwater explorers",
    23: "Neuroscientists, cognitive researchers, and brain science experts",
    24: "Architects, urban planners, and city designers shaping built environments"
}

# ==============================================================================
# TRAIN KMeans + KNN MODEL (k=25)
# ==============================================================================

def build_speaker_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all talks per speaker into one profile."""
    print("📊 Building speaker profiles...")
    
    df['description'] = df['description'].fillna('').astype(str)
    df['title'] = df['title'].fillna('').astype(str)
    df['tags'] = df['tags'].fillna('').astype(str)
    df['speaker_occupation'] = df['speaker_occupation'].fillna('').astype(str)
    
    df['combined_text'] = (
        df['description'] + ' ' + 
        df['title'] + ' ' + 
        df['tags'] + ' ' + 
        df['speaker_occupation']
    )
    
    speaker_profiles = df.groupby('main_speaker').agg({
        'combined_text': lambda x: ' '.join(x),
        'title': 'count',
        'views': 'sum',
        'speaker_occupation': 'first'
    }).reset_index()
    
    speaker_profiles = speaker_profiles.rename(columns={
        'combined_text': 'profile_text',
        'title': 'num_talks',
        'views': 'total_views'
    })
    
    print(f"✅ Created profiles for {len(speaker_profiles)} speakers\n")
    return speaker_profiles


def train_kmeans_knn_model(speaker_profiles: pd.DataFrame,
                           n_clusters: int = 25,
                           max_features: int = 3000,
                           knn_neighbors: int = 5) -> Dict:
    """Train KMeans + KNN pipeline."""
    print(f"🔧 Training KMeans + KNN with k={n_clusters}...\n")
    
    # TF-IDF Vectorization
    print("   [1/3] TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    X_tfidf = vectorizer.fit_transform(speaker_profiles['profile_text'])
    X_dense = X_tfidf.toarray()
    print(f"         Shape: {X_tfidf.shape}")
    
    # KMeans Clustering
    print(f"\n   [2/3] KMeans Clustering...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20,
        max_iter=300
    )
    cluster_labels = kmeans.fit_predict(X_dense)
    speaker_profiles['cluster'] = cluster_labels
    
    # Quality metrics
    sil_score = silhouette_score(X_dense, cluster_labels, sample_size=1000)
    ch_score = calinski_harabasz_score(X_dense, cluster_labels)
    db_score = davies_bouldin_score(X_dense, cluster_labels)
    
    print(f"         Silhouette Score: {sil_score:.4f}")
    print(f"         Calinski-Harabasz: {ch_score:.2f}")
    print(f"         Davies-Bouldin: {db_score:.4f}")
    
    # KNN Classifier
    print(f"\n   [3/3] Training KNN Classifier...")
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors, metric='cosine')
    knn.fit(X_dense, cluster_labels)
    print(f"         ✅ Complete!\n")
    
    # Cluster distribution
    cluster_counts = speaker_profiles['cluster'].value_counts().sort_index()
    print(f"📊 Cluster Size Stats:")
    print(f"   Average: {cluster_counts.mean():.1f} speakers")
    print(f"   Min: {cluster_counts.min()}, Max: {cluster_counts.max()}\n")
    
    return {
        'speaker_profiles': speaker_profiles,
        'vectorizer': vectorizer,
        'kmeans': kmeans,
        'knn': knn,
        'X_dense': X_dense,
        'metrics': {
            'silhouette': sil_score,
            'calinski_harabasz': ch_score,
            'davies_bouldin': db_score
        }
    }


def analyze_clusters(model_data: Dict, top_n_words: int = 10) -> Dict:
    """Extract top keywords for each cluster with names."""
    print("="*80)
    print("📋 CLUSTER ANALYSIS (WITH NAMES)")
    print("="*80 + "\n")
    
    vectorizer = model_data['vectorizer']
    kmeans = model_data['kmeans']
    speaker_profiles = model_data['speaker_profiles']
    
    feature_names = np.array(vectorizer.get_feature_names_out())
    cluster_info = {}
    
    for cluster_id in range(kmeans.n_clusters):
        center = kmeans.cluster_centers_[cluster_id]
        top_indices = center.argsort()[::-1][:top_n_words]
        keywords = feature_names[top_indices].tolist()
        
        cluster_speakers = speaker_profiles[speaker_profiles['cluster'] == cluster_id]
        
        if len(cluster_speakers) > 0:
            top_speakers = cluster_speakers.nlargest(5, 'total_views')['main_speaker'].tolist()
        else:
            top_speakers = []
        
        cluster_info[cluster_id] = {
            'name': CLUSTER_NAMES[cluster_id],
            'category': CLUSTER_CATEGORIES[cluster_id],
            'description': CLUSTER_DESCRIPTIONS[cluster_id],
            'keywords': keywords,
            'top_speakers': top_speakers,
            'size': len(cluster_speakers)
        }
        
        print(f"🏷️  Cluster {cluster_id:2d}: {CLUSTER_NAMES[cluster_id]}")
        print(f"   📂 Category: {CLUSTER_CATEGORIES[cluster_id]}")
        print(f"   👥 Size: {len(cluster_speakers)} speakers")
        print(f"   🔑 Keywords: {', '.join(keywords[:8])}")
        if top_speakers:
            print(f"   ⭐ Top: {', '.join(top_speakers[:3])}")
        print()
    
    return cluster_info


def plot_cluster_visualization(model_data: Dict, 
                               save_path: str = 'models/cluster_visualization.png'):
    """Create PCA 2D visualization with cluster names."""
    print("📊 Creating visualization...")
    
    X_dense = model_data['X_dense']
    labels = model_data['speaker_profiles']['cluster'].values
    speaker_names = model_data['speaker_profiles']['main_speaker'].values
    n_clusters = model_data['kmeans'].n_clusters
    
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_dense)
    
    fig, ax = plt.subplots(figsize=(18, 14))
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_name = CLUSTER_NAMES[cluster_id]
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                  c=[colors[cluster_id]], 
                  label=f'{cluster_id}: {cluster_name}',
                  alpha=0.6, s=60, edgecolors='white', linewidth=0.3)
    
    for i in range(min(30, len(speaker_names))):
        ax.annotate(speaker_names[i], (X_2d[i, 0], X_2d[i, 1]),
                   fontsize=7, alpha=0.6)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                  fontsize=12, fontweight='bold')
    ax.set_title(f'TED Speaker Clusters (k={n_clusters}) - Named Categories', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize=7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}\n")
    plt.close()


def plot_cluster_sizes(model_data: Dict, 
                       save_path: str = 'models/cluster_sizes.png'):
    """Bar chart of cluster sizes."""
    print("📊 Creating cluster size chart...")
    
    cluster_counts = model_data['speaker_profiles']['cluster'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
    bars = ax.bar(cluster_counts.index, cluster_counts.values, color=colors, 
                  edgecolor='black', linewidth=0.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Speakers', fontsize=12, fontweight='bold')
    ax.set_title(f'Cluster Size Distribution (k={len(cluster_counts)})', 
                 fontsize=14, fontweight='bold')
    ax.axhline(y=cluster_counts.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Average: {cluster_counts.mean():.1f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}\n")
    plt.close()


def save_models(model_data: Dict, cluster_info: Dict, output_dir: str = 'models'):
    """Save all models and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(model_data['vectorizer'], f'{output_dir}/vectorizer.joblib')
    joblib.dump(model_data['kmeans'], f'{output_dir}/kmeans.joblib')
    joblib.dump(model_data['knn'], f'{output_dir}/knn.joblib')
    joblib.dump(cluster_info, f'{output_dir}/cluster_info.joblib')
    
    # Save cluster names mapping
    joblib.dump({
        'names': CLUSTER_NAMES,
        'categories': CLUSTER_CATEGORIES,
        'descriptions': CLUSTER_DESCRIPTIONS
    }, f'{output_dir}/cluster_names.joblib')
    
    model_data['speaker_profiles'].to_csv(f'{output_dir}/speaker_clusters.csv', index=False)
    
    with open(f'{output_dir}/metrics.txt', 'w') as f:
        f.write("Clustering Quality Metrics (k=25)\n")
        f.write("="*50 + "\n\n")
        for metric, value in model_data['metrics'].items():
            f.write(f"{metric}: {value}\n")
    
    print(f"💾 Models saved to '{output_dir}/'\n")


def main():
    CSV_PATH = r"C:/Users/Admin/Desktop/TY Btech Sem 5/AIML-Speaker/ted_main.csv"
    N_CLUSTERS = 25
    OUTPUT_DIR = 'models'
    
    print("="*80)
    print("🎤 TED SPEAKER CLUSTERING: KMeans + KNN (k=25) - WITH NAMED CLUSTERS")
    print("="*80 + "\n")
    
    # Load data
    print(f"📁 Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"   {len(df)} talks, {df['main_speaker'].nunique()} speakers\n")
    
    # Build profiles
    speaker_profiles = build_speaker_profiles(df)
    
    # Train model
    model_data = train_kmeans_knn_model(speaker_profiles, n_clusters=N_CLUSTERS)
    
    # Analyze clusters
    cluster_info = analyze_clusters(model_data)
    
    # Create plots
    print("="*80)
    print("📊 CREATING PLOTS")
    print("="*80 + "\n")
    plot_cluster_sizes(model_data, f'{OUTPUT_DIR}/cluster_sizes.png')
    plot_cluster_visualization(model_data, f'{OUTPUT_DIR}/cluster_visualization.png')
    
    # Save everything
    save_models(model_data, cluster_info, OUTPUT_DIR)
    
    print("="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📂 Saved to: {OUTPUT_DIR}/")
    print(f"   - vectorizer.joblib")
    print(f"   - kmeans.joblib")
    print(f"   - knn.joblib")
    print(f"   - cluster_info.joblib")
    print(f"   - cluster_names.joblib (NEW!)")
    print(f"   - speaker_clusters.csv")
    print(f"\n🧪 Run test.py to see confidence scores!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()