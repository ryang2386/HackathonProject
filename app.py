from fastapi import FastAPI
from pydantic import BaseModel, Field
import requests
import csv
import json

app = FastAPI()

# Initialize empty properties list
properties = []

def _find_next_id():
    if not properties:
        return 1
    return max(property.unique_id for property in properties) + 1

class Property(BaseModel):
    unique_id: int
    property_address: str
    floor: str
    suite: int
    size_SF: int
    rent_SF_year: float
    associate_1: str
    broker_email_id: str
    associate_2: str
    associate_3: str
    associate_4: str
    annual_rent: float
    monthly_rent: float
    gci_on_3_years: float

# Add initial properties
properties.extend([
    Property(unique_id=1, property_address="123 Main St", floor="1st", suite=101, size_SF=1000, rent_SF_year=20.0,
             associate_1="John Doe", broker_email_id="john.doe@example.com", associate_2="Jane Smith",
             associate_3="Mike Johnson", associate_4="Emily Davis", annual_rent=20000.0, monthly_rent=1666.67,
             gci_on_3_years=6000.0),
    Property(unique_id=2, property_address="456 Elm St", floor="2nd", suite=202, size_SF=1500, rent_SF_year=25.0,
             associate_1="Alice Brown", broker_email_id="alice.brown@example.com", associate_2="Bob White",
             associate_3="Charlie Black", associate_4="Diana Green", annual_rent=37500.0, monthly_rent=3125.0,
             gci_on_3_years=11250.0)
])

def import_properties_from_csv(csv_file_path: str):
    """
    Import properties from a CSV file and add them to the properties list
    
    Expected CSV columns:
    property_address, floor, suite, size_SF, rent_SF_year, associate_1, 
    broker_email_id, associate_2, associate_3, associate_4, annual_rent, 
    monthly_rent, gci_on_3_years
    """
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            imported_count = 0
            for row in reader:
                try:
                    # Handle currency formatting and convert string values to appropriate types
                    def clean_currency(value):
                        """Remove $ signs, commas, and quotes from currency values"""
                        if isinstance(value, str):
                            return value.replace('$', '').replace(',', '').replace('"', '').strip()
                        return value
                    
                    def clean_number(value):
                        """Clean and convert numeric values"""
                        cleaned = clean_currency(value)
                        return float(cleaned) if cleaned else 0.0
                    
                    # Map your CSV columns to the expected format
                    property_data = {
                        'unique_id': int(row['unique_id']),
                        'property_address': row['Property Address'],
                        'floor': row['Floor'],
                        'suite': int(row['Suite']),
                        'size_SF': int(row['Size (SF)']),
                        'rent_SF_year': clean_number(row['Rent/SF/Year']),
                        'associate_1': row['Associate 1'],
                        'broker_email_id': row['BROKER Email ID'],
                        'associate_2': row['Associate 2'],
                        'associate_3': row['Associate 3'],
                        'associate_4': row['Associate 4'],
                        'annual_rent': clean_number(row['Annual Rent']),
                        'monthly_rent': clean_number(row['Monthly Rent']),
                        'gci_on_3_years': clean_number(row['GCI On 3 Years'])
                    }
                    
                    # Create Property object and add to list
                    new_property = Property(**property_data)
                    properties.append(new_property)
                    imported_count += 1
                    
                    print(f"Imported: {property_data['property_address']}")
                    
                except (ValueError, KeyError) as e:
                    print(f"Error importing row {reader.line_num}: {e}")
                    continue
                    
            print(f"Successfully imported {imported_count} properties from {csv_file_path}")
            return imported_count
            
    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
        return 0
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return 0

@app.get("/properties")
async def get_properties():
    return properties

@app.post("/properties", status_code=201)
async def create_property(property: Property):
    properties.append(property)
    return property

@app.post("/upload_docs")
async def upload_docs(csv_file_path: str):
    """Import properties from a CSV file"""
    count = import_properties_from_csv(csv_file_path)
    return {"message": f"Imported {count} properties from CSV", "total_properties": len(properties)}