import numpy as np
import joblib
from typing import Dict

# ==============================================================================
# TEST SPEAKER PREDICTION WITH CONFIDENCE SCORES & CLUSTER NAMES
# ==============================================================================

def load_models(model_dir: str = 'models') -> Dict:
    """Load trained models including cluster names."""
    print("📦 Loading models...")
    models = {
        'vectorizer': joblib.load(f'{model_dir}/vectorizer.joblib'),
        'kmeans': joblib.load(f'{model_dir}/kmeans.joblib'),
        'knn': joblib.load(f'{model_dir}/knn.joblib'),
        'cluster_info': joblib.load(f'{model_dir}/cluster_info.joblib'),
        'cluster_names': joblib.load(f'{model_dir}/cluster_names.joblib')
    }
    print(f"✅ Loaded models with {models['kmeans'].n_clusters} named clusters\n")
    return models


def predict_speaker_with_confidence(speaker_data: Dict, models: Dict) -> Dict:
    """
    Predict cluster for new speaker with improved confidence calculation.
    Uses softmax on inverse distances for probability distribution.
    """
    # Combine text
    text_parts = [
        speaker_data.get('description', ''),
        speaker_data.get('title', ''),
        speaker_data.get('tags', ''),
        speaker_data.get('speaker_occupation', '')
    ]
    combined_text = ' '.join([str(t) for t in text_parts if t])
    
    # Vectorize
    vec = models['vectorizer'].transform([combined_text])
    vec_dense = vec.toarray()
    
    # Predict using KMeans
    predicted_cluster = int(models['kmeans'].predict(vec)[0])
    
    # Calculate distances to ALL cluster centers
    centers = models['kmeans'].cluster_centers_
    distances = np.linalg.norm(centers - vec_dense, axis=1)
    
    # Improved confidence: Softmax-based probability
    similarities = 1 / (1 + distances)
    confidence = similarities[predicted_cluster] / similarities.sum()
    
    # Alternative: Relative distance confidence
    min_dist = distances[predicted_cluster]
    second_min_dist = np.partition(distances, 1)[1]
    relative_confidence = 1 - (min_dist / second_min_dist) if second_min_dist > 0 else 1.0
    
    # Get cluster info
    cluster_data = models['cluster_info'][predicted_cluster]
    cluster_names_data = models['cluster_names']
    
    # Find top 3 closest clusters
    closest_clusters_ids = distances.argsort()[:3]
    closest_clusters = [
        {
            'id': int(cid),
            'name': cluster_names_data['names'][int(cid)],
            'category': cluster_names_data['categories'][int(cid)],
            'distance': float(distances[cid])
        }
        for cid in closest_clusters_ids
    ]
    
    result = {
        'predicted_cluster': predicted_cluster,
        'cluster_name': cluster_data['name'],
        'cluster_category': cluster_data['category'],
        'cluster_description': cluster_data['description'],
        'confidence': float(confidence),
        'relative_confidence': float(relative_confidence),
        'distance_to_center': float(min_dist),
        'closest_clusters': closest_clusters,
        'cluster_info': {
            'keywords': cluster_data['keywords'][:10],
            'top_speakers': cluster_data['top_speakers'][:5],
            'size': cluster_data['size']
        }
    }
    
    return result


def print_prediction_result(speaker_name: str, result: Dict):
    """Print prediction results with cluster names."""
    print("="*80)
    print(f"🎤 PREDICTION: {speaker_name}")
    print("="*80)
    
    print(f"\n✅ Predicted Cluster #{result['predicted_cluster']}: {result['cluster_name']}")
    print(f"   📂 Category: {result['cluster_category']}")
    print(f"   👥 Cluster Size: {result['cluster_info']['size']} speakers")
    print(f"   📝 {result['cluster_description']}")
    
    print(f"\n📊 Confidence Score: {result['confidence']:.1%}")
    print(f"   Alternative (Relative): {result['relative_confidence']:.1%}")
    print(f"   Distance to Center: {result['distance_to_center']:.4f}")
    
    print(f"\n🏷️  Cluster Keywords:")
    print(f"   {', '.join(result['cluster_info']['keywords'][:8])}")
    
    print(f"\n👥 Similar Speakers in this Cluster:")
    for speaker in result['cluster_info']['top_speakers'][:3]:
        print(f"   • {speaker}")
    
    print(f"\n📍 Alternative Cluster Matches:")
    for i, cluster in enumerate(result['closest_clusters'][:3], 1):
        print(f"   {i}. {cluster['name']} ({cluster['category']}) - Distance: {cluster['distance']:.4f}")
    
    print("\n" + "="*80 + "\n")


def test_multiple_speakers(models: Dict):
    """Test on multiple diverse speaker profiles."""
    
    test_cases = [
        {
            'name': 'AI Healthcare Expert',
            'data': {
                'description': 'Exploring how artificial intelligence and machine learning are revolutionizing healthcare through early disease detection, personalized treatment plans, and predictive analytics.',
                'title': 'AI Transforming Medicine',
                'tags': 'AI, healthcare, machine learning, technology, medicine, data science',
                'speaker_occupation': 'AI Researcher & Physician'
            }
        },
        {
            'name': 'Climate Activist',
            'data': {
                'description': 'Urgent call to action on climate change, discussing renewable energy solutions, carbon footprint reduction, and sustainable practices for a better planet.',
                'title': 'Saving Our Planet',
                'tags': 'climate change, environment, sustainability, renewable energy, activism',
                'speaker_occupation': 'Environmental Scientist'
            }
        },
        {
            'name': 'Music Performer',
            'data': {
                'description': 'A mesmerizing live musical performance combining beatboxing, vocal loops, and comedy to create unique soundscapes that blur the boundaries of genre.',
                'title': 'The Art of Sound',
                'tags': 'music, performance, live music, entertainment, creativity, comedy',
                'speaker_occupation': 'Musician & Performer'
            }
        },
        {
            'name': 'Neuroscience Researcher',
            'data': {
                'description': 'Fascinating journey into the human brain, exploring consciousness, memory formation, neuroplasticity, and how our thoughts shape our reality.',
                'title': 'Inside the Mind',
                'tags': 'brain, neuroscience, psychology, cognition, science, mind',
                'speaker_occupation': 'Neuroscientist'
            }
        },
        {
            'name': 'Social Entrepreneur',
            'data': {
                'description': 'Building businesses that create positive social impact, addressing poverty, inequality, and empowering communities through innovative solutions.',
                'title': 'Business for Good',
                'tags': 'business, social enterprise, impact, entrepreneurship, development',
                'speaker_occupation': 'Social Entrepreneur'
            }
        },
        {
            'name': 'Education Reformer',
            'data': {
                'description': 'Reimagining education for the 21st century, discussing creativity in schools, personalized learning, and preparing students for an uncertain future.',
                'title': 'Schools Kill Creativity',
                'tags': 'education, learning, creativity, schools, teaching, reform',
                'speaker_occupation': 'Education Expert'
            }
        },
        {
            'name': 'Marine Biologist',
            'data': {
                'description': 'Diving deep into ocean conservation, coral reef protection, and the urgent need to save our marine ecosystems from pollution and overfishing.',
                'title': 'Saving Our Oceans',
                'tags': 'ocean, marine biology, conservation, environment, wildlife, coral reefs',
                'speaker_occupation': 'Marine Biologist'
            }
        }
    ]
    
    print("\n" + "="*80)
    print("🧪 TESTING SPEAKER PREDICTIONS")
    print("="*80 + "\n")
    
    results = []
    for test_case in test_cases:
        result = predict_speaker_with_confidence(test_case['data'], models)
        results.append((test_case['name'], result))
        print_prediction_result(test_case['name'], result)
    
    # Summary table
    print("="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80 + "\n")
    
    print(f"{'Speaker':<30} {'Cluster':<10} {'Name':<35} {'Confidence':<12}")
    print("-"*100)
    for name, result in results:
        cluster_label = f"#{result['predicted_cluster']}"
        print(f"{name:<30} {cluster_label:<10} {result['cluster_name']:<35} {result['confidence']:<12.1%}")
    
    print("\n" + "="*80 + "\n")


def main():
    print("\n" + "="*80)
    print("🎤 TED SPEAKER PREDICTION TEST (k=25) - WITH CLUSTER NAMES")
    print("="*80 + "\n")
    
    # Load models
    models = load_models('models')
    
    # Test on multiple speakers
    test_multiple_speakers(models)
    
    print("="*80)
    print("✅ TESTING COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()