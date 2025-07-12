import requests

# Test data for new property
new_property = {
    "property_address": "789 Oak St",
    "floor": "3rd",
    "suite": 303,
    "size_SF": 1200,
    "rent_SF_year": 22.0,
    "associate_1": "Tom Blue",
    "broker_email_id": "tom.blue@example.com",
    "associate_2": "Sara White",
    "associate_3": "James Black",
    "associate_4": "Linda Green",
    "annual_rent": 26400.0,
    "monthly_rent": 2200.0,
    "gci_on_3_years": 7920.0
}

def test_create_property():
    """Test creating a new property via POST request"""
    try:
        response = requests.post("http://localhost:8000/properties", json=new_property)
        
        if response.status_code == 201:
            print("✅ Property created successfully!")
            print("Response:", response.json())
        else:
            print(f"❌ Failed to create property. Status code: {response.status_code}")
            print("Response:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the FastAPI server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_get_properties():
    """Test getting all properties via GET request"""
    try:
        response = requests.get("http://localhost:8000/properties")
        
        if response.status_code == 200:
            properties = response.json()
            print(f"✅ Retrieved {len(properties)} properties:")
            for prop in properties:
                print(f"  - ID {prop['unique_id']}: {prop['property_address']}")
        else:
            print(f"❌ Failed to get properties. Status code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the FastAPI server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🏢 Testing Property API")
    print("=" * 40)
    
    print("\n1. Getting existing properties:")
    test_get_properties()
    
    print("\n2. Creating new property:")
    test_create_property()
    
    print("\n3. Getting properties after creation:")
    test_get_properties()
