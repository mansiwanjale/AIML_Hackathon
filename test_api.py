"""
Test script to verify the backend API is working correctly
Run this AFTER starting backend.py
"""

import requests
import json

API_BASE = "http://localhost:5000/api"

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_health():
    print_section("1. TESTING HEALTH CHECK")
    try:
        response = requests.get(f"{API_BASE}/health")
        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"📊 Total Talks: {data['total_talks']}")
        print(f"👥 Total Speakers: {data['total_speakers']}")
        print(f"🏷️  Total Categories: {data['total_categories']}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_categories():
    print_section("2. TESTING CATEGORIES ENDPOINT")
    try:
        response = requests.get(f"{API_BASE}/categories")
        categories = response.json()
        print(f"✅ Found {len(categories)} categories")
        print(f"\nFirst 3 categories:")
        for cat in categories[:3]:
            print(f"   #{cat['id']}: {cat['name']} ({cat['speaker_count']} speakers)")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_category_detail():
    print_section("3. TESTING CATEGORY DETAIL")
    category_id = 12  # Cancer & Medical Research
    try:
        response = requests.get(f"{API_BASE}/categories/{category_id}")
        data = response.json()
        print(f"✅ Category #{data['id']}: {data['name']}")
        print(f"📝 Description: {data['description'][:100]}...")
        print(f"👥 Speakers: {data['speaker_count']}")
        print(f"🎤 Total Talks: {data['total_talks']}")
        print(f"🔑 Keywords: {', '.join(data['keywords'][:5])}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_search():
    print_section("4. TESTING SEARCH")
    query = "technology"
    try:
        response = requests.get(f"{API_BASE}/search?q={query}")
        results = response.json()
        print(f"✅ Found {len(results)} results for '{query}'")
        for result in results[:3]:
            print(f"   • {result['name']} ({result['speaker_count']} speakers)")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_speaker():
    print_section("5. TESTING SPEAKER ENDPOINT")
    speaker_name = "Ken Robinson"
    try:
        response = requests.get(f"{API_BASE}/speakers/{speaker_name}")
        data = response.json()
        print(f"✅ Speaker: {data['speaker_name']}")
        print(f"💼 Occupation: {data['occupation']}")
        print(f"🏷️  Category: {data['category_name']}")
        print(f"🎤 Total Talks: {data['total_talks']}")
        print(f"👁️  Total Views: {data['total_views']:,}")
        if data['talks']:
            print(f"\nFirst talk:")
            print(f"   Title: {data['talks'][0]['title']}")
            print(f"   URL: {data['talks'][0]['url']}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_analyze():
    print_section("6. TESTING ANALYZER")
    
    test_speaker = {
        "name": "Test Speaker",
        "occupation": "Neuroscientist",
        "title": "The Brain's Hidden Potential",
        "description": "Exploring how our brain works, neuroplasticity, and consciousness",
        "tags": "brain, neuroscience, psychology, science"
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/analyze",
            json=test_speaker,
            headers={'Content-Type': 'application/json'}
        )
        data = response.json()
        print(f"✅ Analysis Complete")
        print(f"🎯 Predicted Category: #{data['predicted_category']['id']} - {data['predicted_category']['name']}")
        print(f"📊 Confidence: {data['confidence']:.1f}%")
        print(f"📂 Category Type: {data['predicted_category']['category']}")
        print(f"🔑 Keywords: {', '.join(data['keywords'][:5])}")
        print(f"\nAlternative Matches:")
        for alt in data['alternative_matches'][:3]:
            print(f"   • {alt['name']} ({alt['confidence']*100:.1f}%)")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_stats():
    print_section("7. TESTING STATS ENDPOINT")
    try:
        response = requests.get(f"{API_BASE}/stats")
        data = response.json()
        print(f"✅ System Statistics")
        print(f"📊 Total Talks: {data['total_talks']}")
        print(f"👥 Total Speakers: {data['total_speakers']}")
        print(f"🏷️  Total Categories: {data['total_categories']}")
        print(f"👁️  Total Views: {data['total_views']:,}")
        print(f"\nTop 5 Largest Categories:")
        sorted_cats = sorted(data['category_distribution'], key=lambda x: x['count'], reverse=True)
        for cat in sorted_cats[:5]:
            print(f"   • {cat['name']}: {cat['count']} speakers")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("\n" + "🚀"*30)
    print("  TEDx SPEAKER INTELLIGENCE - API TEST SUITE")
    print("🚀"*30)
    
    print("\n⚠️  Make sure backend.py is running on http://localhost:5000")
    input("Press Enter to start tests...")
    
    results = []
    
    results.append(("Health Check", test_health()))
    results.append(("Categories List", test_categories()))
    results.append(("Category Detail", test_category_detail()))
    results.append(("Search", test_search()))
    results.append(("Speaker Info", test_speaker()))
    results.append(("Analyzer", test_analyze()))
    results.append(("Statistics", test_stats()))
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'='*60}\n")
    
    if passed == total:
        print("🎉 All tests passed! Your backend is working perfectly!")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    print("\n✨ Next step: Open index.html in your browser")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()