from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
import csv
import os
import requests
import json

app = FastAPI()

load_dotenv()  # Load environment variables from .env file

# Initialize empty properties list
properties = []

# Implement OpenAI API key
class ConversationalOpenAI:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.conversation_history = []

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.conversation_history.append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})

    def chat(self, user_message: str, model: str ="gpt-3.5-turbo", max_tokens: int = 150):
        self.add_message("user", user_message)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = [{"role": msg["role"], "content": msg["content"]} for msg in self.conversation_history]

        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=data)

            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content']
                self.add_message("assistant", ai_response)

                return {
                    "success": True,
                    "response": ai_response,
                    "tokens_used": result['usage']['total_tokens']
                }
            else:
                return {
                    "success": False,
                    "error": f"API request failed with status code {response.status_code}",
                    "details": response.text
                }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Request is no good: {str(e)}"
            }
        
    def get_saved_conversation(self):
        """Return the saved conversation history"""
        return self.conversation_history
    
    def save_conversation(self, filename="conversation_history.json"):
        """Save the conversation history to a JSON file"""
        with open(filename, "w") as f:
            json.dump(self.conversation_history, f, indent=1)
        print(f"Conversation is saved! Check {filename} for details.")
    
    def clear_conversation(self):
        """Clear the conversation history"""
        self.conversation_history = []
        print("Conversation history cleared.")

def ai_user_chat():
    print("AI Chat Ready!")
    print ("Type 'quit' to exit or 'save' to save the conversation.")
    print ("To clear the conversation, type 'clear'.")
    user_message = input("Type your message here then press Enter:")

    try:
        ai = ConversationalOpenAI()
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    while True:
        if user_message.lower() == "quit":
            print("Exiting chat. Goodbye!")
            break
        elif user_message.lower() == "save":
            ai.save_conversation()
            print("Conversation saved successfully.")
        elif user_message.lower() == "clear":
            ai.clear_conversation()
        else:
            response = ai.chat(user_message)
            if response["success"]:
                print(f"AI: {response['response']}")
                print(f"Tokens used: {response['tokens_used']}")
            else:
                print(f"Error: {response['error']}")
        


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