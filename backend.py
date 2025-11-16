from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# ==============================================================================
# LOAD MODELS AND DATA
# ==============================================================================

MODEL_DIR = 'models'
CSV_PATH = os.getenv('CSV_PATH', 'ted_main.csv')

print("🔄 Loading models and data...")

# Load models
vectorizer = joblib.load(f'{MODEL_DIR}/vectorizer.joblib')
kmeans = joblib.load(f'{MODEL_DIR}/kmeans.joblib')
knn = joblib.load(f'{MODEL_DIR}/knn.joblib')
cluster_info = joblib.load(f'{MODEL_DIR}/cluster_info.joblib')
cluster_names = joblib.load(f'{MODEL_DIR}/cluster_names.joblib')

# Load CSV data
df = pd.read_csv(CSV_PATH)

# Clean data
df['description'] = df['description'].fillna('').astype(str)
df['title'] = df['title'].fillna('').astype(str)
df['tags'] = df['tags'].fillna('').astype(str)
df['speaker_occupation'] = df['speaker_occupation'].fillna('').astype(str)
df['main_speaker'] = df['main_speaker'].fillna('Unknown').astype(str)
df['url'] = df['url'].fillna('').astype(str)
df['views'] = df['views'].fillna(0).astype(int)

# Load speaker clusters
speaker_clusters_df = pd.read_csv(f'{MODEL_DIR}/speaker_clusters.csv')

# Create speaker to cluster mapping
speaker_to_cluster = dict(zip(speaker_clusters_df['main_speaker'], speaker_clusters_df['cluster']))

# Add cluster info to main dataframe
df['cluster'] = df['main_speaker'].map(speaker_to_cluster).fillna(-1).astype(int)

print(f"✅ Loaded {len(df)} talks from {df['main_speaker'].nunique()} speakers")
print(f"✅ Models ready with {len(cluster_info)} categories\n")

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'total_talks': len(df),
        'total_speakers': df['main_speaker'].nunique(),
        'total_categories': len(cluster_info)
    })


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all categories with metadata"""
    categories = []
    
    for cluster_id, info in cluster_info.items():
        # Count speakers in this category
        speaker_count = len(df[df['cluster'] == cluster_id]['main_speaker'].unique())
        
        categories.append({
            'id': cluster_id,
            'name': info['name'],
            'category': info['category'],
            'description': info['description'],
            'keywords': info['keywords'][:8],
            'speaker_count': speaker_count,
            'top_speakers': info['top_speakers'][:3]
        })
    
    return jsonify(categories)


@app.route('/api/categories/<int:category_id>', methods=['GET'])
def get_category_detail(category_id):
    """Get detailed information about a specific category"""
    if category_id not in cluster_info:
        return jsonify({'error': 'Category not found'}), 404
    
    info = cluster_info[category_id]
    
    # Get all speakers in this category
    category_df = df[df['cluster'] == category_id]
    speakers_in_category = category_df['main_speaker'].unique().tolist()
    
    # Get talk count per speaker
    speaker_talk_counts = category_df.groupby('main_speaker').size().to_dict()
    
    return jsonify({
        'id': category_id,
        'name': info['name'],
        'category': info['category'],
        'description': info['description'],
        'keywords': info['keywords'],
        'speaker_count': len(speakers_in_category),
        'total_talks': len(category_df),
        'top_speakers': info['top_speakers'],
        'all_speakers': speakers_in_category,
        'speaker_talk_counts': speaker_talk_counts
    })


@app.route('/api/speakers/<speaker_name>', methods=['GET'])
def get_speaker_talks(speaker_name):
    """Get all talks by a specific speaker"""
    speaker_df = df[df['main_speaker'] == speaker_name]
    
    if len(speaker_df) == 0:
        return jsonify({'error': 'Speaker not found'}), 404
    
    talks = []
    for _, row in speaker_df.iterrows():
        talks.append({
            'title': row['title'],
            'description': row['description'],
            'tags': row['tags'],
            'url': row['url'],
            'views': int(row['views']),
            'event': row['event'] if 'event' in row else 'TED',
            'duration': int(row['duration']) if 'duration' in row and pd.notna(row['duration']) else 0
        })
    
    # Get speaker's category
    speaker_cluster = speaker_to_cluster.get(speaker_name, -1)
    category_name = cluster_info[speaker_cluster]['name'] if speaker_cluster in cluster_info else 'Unknown'
    
    return jsonify({
        'speaker_name': speaker_name,
        'occupation': speaker_df.iloc[0]['speaker_occupation'],
        'category_id': int(speaker_cluster),
        'category_name': category_name,
        'total_talks': len(talks),
        'total_views': int(speaker_df['views'].sum()),
        'talks': talks
    })


@app.route('/api/search', methods=['GET'])
def search_categories():
    """Search categories by name or keywords"""
    query = request.args.get('q', '').lower()
    
    if not query:
        return jsonify([])
    
    results = []
    for cluster_id, info in cluster_info.items():
        # Search in name, category, keywords
        searchable_text = (
            info['name'].lower() + ' ' +
            info['category'].lower() + ' ' +
            ' '.join(info['keywords']).lower()
        )
        
        if query in searchable_text:
            speaker_count = len(df[df['cluster'] == cluster_id]['main_speaker'].unique())
            results.append({
                'id': cluster_id,
                'name': info['name'],
                'category': info['category'],
                'description': info['description'],
                'keywords': info['keywords'][:5],
                'speaker_count': speaker_count
            })
    
    return jsonify(results)


@app.route('/api/analyze', methods=['POST'])
def analyze_speaker():
    """Analyze speaker profile and predict category"""
    data = request.json
    
    # Get input fields (all optional)
    description = data.get('description', '')
    title = data.get('title', '')
    tags = data.get('tags', '')
    occupation = data.get('occupation', '')
    name = data.get('name', '')
    
    # Combine available text
    text_parts = [description, title, tags, occupation]
    combined_text = ' '.join([str(t) for t in text_parts if t])
    
    if not combined_text.strip():
        return jsonify({'error': 'Please provide at least one field'}), 400
    
    # Vectorize
    vec = vectorizer.transform([combined_text])
    vec_dense = vec.toarray()
    
    # Predict using KMeans
    predicted_cluster = int(kmeans.predict(vec)[0])
    
    # Calculate distances to ALL cluster centers
    centers = kmeans.cluster_centers_
    distances = np.linalg.norm(centers - vec_dense, axis=1)
    
    # Calculate confidence
    similarities = 1 / (1 + distances)
    confidence = similarities[predicted_cluster] / similarities.sum()
    
    # Get alternative matches (top 3 closest)
    closest_clusters_ids = distances.argsort()[:3]
    alternative_matches = [
        {
            'id': int(cid),
            'name': cluster_info[int(cid)]['name'],
            'category': cluster_info[int(cid)]['category'],
            'confidence': float(similarities[cid] / similarities.sum()),
            'distance': float(distances[cid])
        }
        for cid in closest_clusters_ids
    ]
    
    # Get cluster info
    cluster_data = cluster_info[predicted_cluster]
    
    result = {
        'speaker_name': name if name else 'Anonymous',
        'predicted_category': {
            'id': predicted_cluster,
            'name': cluster_data['name'],
            'category': cluster_data['category'],
            'description': cluster_data['description']
        },
        'confidence': float(confidence * 100),  # Convert to percentage
        'distance': float(distances[predicted_cluster]),
        'keywords': cluster_data['keywords'][:10],
        'similar_speakers': cluster_data['top_speakers'][:5],
        'alternative_matches': alternative_matches,
        'cluster_size': cluster_data['size']
    }
    
    return jsonify(result)


@app.route('/api/stats', methods=['GET'])
def get_system_stats():
    """Get system statistics"""
    category_sizes = df['cluster'].value_counts().to_dict()
    
    # Category distribution
    category_distribution = []
    for cluster_id in sorted(cluster_info.keys()):
        category_distribution.append({
            'id': cluster_id,
            'name': cluster_info[cluster_id]['name'],
            'count': category_sizes.get(cluster_id, 0)
        })
    
    return jsonify({
        'total_talks': len(df),
        'total_speakers': df['main_speaker'].nunique(),
        'total_categories': len(cluster_info),
        'total_views': int(df['views'].sum()),
        'category_distribution': category_distribution,
        'model_info': {
            'algorithm': 'KMeans + KNN',
            'n_clusters': len(cluster_info),
            'features': vectorizer.max_features
        }
    })


# ==============================================================================
# RUN SERVER
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 TEDx Speaker Intelligence API Server")
    print("="*60)
    print(f"📊 Loaded {len(df)} talks")
    print(f"👥 {df['main_speaker'].nunique()} speakers")
    print(f"🏷️  {len(cluster_info)} categories")
    print("\n🌐 Server running on: http://localhost:5000")
    print("📡 API Documentation:")
    print("   GET  /api/health")
    print("   GET  /api/categories")
    print("   GET  /api/categories/<id>")
    print("   GET  /api/speakers/<name>")
    print("   GET  /api/search?q=<query>")
    print("   POST /api/analyze")
    print("   GET  /api/stats")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)