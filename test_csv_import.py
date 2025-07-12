"""
Test script for CSV import and JSON export functionality
Run this after starting your FastAPI server
"""

import requests

def test_csv_import():
    """Test importing properties from CSV file"""
    print("🏢 Testing CSV Import")
    print("=" * 40)
    
    # Get current properties count
    response = requests.get("http://localhost:8000/properties")
    if response.status_code == 200:
        initial_count = len(response.json())
        print(f"Initial properties count: {initial_count}")
    
    # Import from CSV
    csv_file_path = "HackathonInternalKnowledgeBase.csv"
    response = requests.post(f"http://localhost:8000/upload_docs?csv_file_path={csv_file_path}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ {result['message']}")
        print(f"Total properties now: {result['total_properties']}")
    else:
        print(f"❌ Import failed: {response.text}")

def test_direct_import():
    """Test the direct import function (not via API)"""
    print("\n🔄 Testing Direct CSV Import")
    print("=" * 40)
    
    # This would be used if you call the function directly in your app
    from app import import_properties_from_csv, export_properties_to_json
    
    count = import_properties_from_csv("HackathonInternalKnowledgeBase.csv")
    print(f"Directly imported {count} properties")
    
    # Export to see all properties
    export_count = export_properties_to_json("direct_export.json")
    print(f"Exported {export_count} properties to direct_export.json")

if __name__ == "__main__":
    try:
        print("🚀 Starting CSV/JSON Tests")
        print("Make sure your FastAPI server is running on http://localhost:8000\n")
        
        test_csv_import()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("\nYou can also:")
        print("1. Check the FastAPI docs at http://localhost:8000/docs")
        print("2. View all properties at http://localhost:8000/properties")
        print("3. Use the HackathonInternalKnowledgeBase.csv file as a template for your own data")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server.")
        print("Please start your FastAPI server first with: python -m uvicorn app:app --reload")
    except Exception as e:
        print(f"❌ Error: {e}")
