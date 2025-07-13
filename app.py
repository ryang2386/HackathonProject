from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
import csv
import os
import requests
import json
import hashlib
import secrets

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

    def chat(self, user_message: str, model: str ="gpt-4o-mini", max_tokens: int = 150):
        # Check if user is asking about properties or CSV
        property_info = ""
        if any(word in user_message.lower() for word in ["property", "properties", "rent", "csv", "export", "document", "how many", "average", "highest", "lowest", "largest", "biggest", "cheapest", "most expensive"]):
            property_info = query_properties(user_message)
            print(f"Property query result: {property_info}")  # Debug print
            
        # Add property context if relevant
        if property_info and "No properties" not in property_info:
            enhanced_message = f"User question: {user_message}\n\nProperty data context: {property_info}\n\nPlease provide a helpful response based on this property information."
            self.add_message("user", enhanced_message)
        else:
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
                error_details = ""
                try:
                    error_json = response.json()
                    error_details = error_json.get("error", {}).get("message", response.text)
                except:
                    error_details = response.text
                
                return {
                    "success": False,
                    "error": f"API request failed with status code {response.status_code}",
                    "details": error_details,
                    "api_key_used": self.api_key[:10] + "..." if self.api_key else "None"
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

# Initialize AI instance globally
ai = ConversationalOpenAI()

# Pydantic models for AI endpoints
class UserChatMessage(BaseModel):
    user_id: str
    message: str
    model: str = "gpt-4o-mini"
    max_tokens: int = 150

class ChatResponse(BaseModel):
    success: bool
    response: str = None
    tokens_used: int = None
    error: str = None

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(chat_message: UserChatMessage):
    """Send a message to the AI and get a response"""
    if chat_message.user_id not in users:
        return ChatResponse(success = False, error = "User does not exist")

    if chat_message.user_id not in user_conversations:
        user_conversations[chat_message.user_id] = ConversationalOpenAI()

    user_ai = user_conversations[chat_message.user_id]

    try:
        result = user_ai.chat(
            user_message=chat_message.message,
            model=chat_message.model,
            max_tokens=chat_message.max_tokens
        )
        return ChatResponse(**result)
    except Exception as e:
        return ChatResponse(
            success=False,
            error=f"Chat error: {str(e)}"
        )

@app.get("/crm/conversations/{user_id}")
async def get_chat_history(user_id: str):
    """Get the conversation history"""
    if user_id not in users:
        return {"User doesn't exist"}
    
    if user_id not in user_conversations:
        return {"error": "User does not exist or has no conversation history"}

    return {"conversation_history": user_conversations[user_id].get_saved_conversation()}

@app.post("/reset/{user_id}")
async def clear_chat_history(user_id: str):
    """Clear the conversation history"""
    if user_id not in users:
        return {"User does not exist."}

    if user_id in user_conversations:
        user_conversations[user_id].clear_conversation()

    return {"message": "Conversation history cleared"}

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

class User(BaseModel):
    user_id: str
    email: str
    password: str

class UserLogin(BaseModel):
    user_id: str
    password: str

class UserRegistration(BaseModel):
    user_id: str
    password: str
    email: str

users = {}
user_conversations = {}

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    return hashlib.sha256((password + salt).encode()).hexdigest() + ":" + salt

def verify_password(password: str, password_hash: str) -> bool:
    try:
        hash_part, salt = password_hash.split(":")
        return hashlib.sha256((password + salt).encode()).hexdigest() == hash_part
    except ValueError:
        return False
    
def auth_user(user_id: str, password: str) -> User:
    user = users.get(user_id)
    if user and verify_password(password, user.password_hash):
        return user
    return None

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
            skipped_count = 0
            error_details = []
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 because row 1 is header
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
                        try:
                            return float(cleaned) if cleaned else 0.0
                        except ValueError:
                            return 0.0
                    
                    def clean_int(value):
                        """Clean and convert to integer"""
                        cleaned = clean_currency(value)
                        try:
                            return int(float(cleaned)) if cleaned else 0
                        except ValueError:
                            return 0
                    
                    # Debug: Print the row being processed
                    print(f"Processing row {row_num}: {row.get('Property Address', 'Unknown')}")
                    
                    # Map your CSV columns to the expected format with better error handling
                    property_data = {
                        'unique_id': clean_int(row.get('unique_id', row_num)),  # Use row number if unique_id missing
                        'property_address': row.get('Property Address', '').strip(),
                        'floor': row.get('Floor', '').strip(),
                        'suite': clean_int(row.get('Suite', 0)),
                        'size_SF': clean_int(row.get('Size (SF)', 0)),
                        'rent_SF_year': clean_number(row.get('Rent/SF/Year', 0)),
                        'associate_1': row.get('Associate 1', '').strip(),
                        'broker_email_id': row.get('BROKER Email ID', '').strip(),
                        'associate_2': row.get('Associate 2', '').strip(),
                        'associate_3': row.get('Associate 3', '').strip(),
                        'associate_4': row.get('Associate 4', '').strip(),
                        'annual_rent': clean_number(row.get('Annual Rent', 0)),
                        'monthly_rent': clean_number(row.get('Monthly Rent', 0)),
                        'gci_on_3_years': clean_number(row.get('GCI On 3 Years', 0))
                    }
                    
                    # Validate essential fields
                    if not property_data['property_address']:
                        skipped_count += 1
                        error_details.append(f"Row {row_num}: Missing property address")
                        continue
                    
                    # Create Property object and add to list
                    new_property = Property(**property_data)
                    properties.append(new_property)
                    imported_count += 1
                    
                    if imported_count % 50 == 0:  # Progress indicator
                        print(f"Imported {imported_count} properties so far...")
                    
                except (ValueError, KeyError, TypeError) as e:
                    skipped_count += 1
                    error_msg = f"Row {row_num}: {str(e)} - Data: {dict(row)}"
                    error_details.append(error_msg)
                    print(f"Error importing row {row_num}: {e}")
                    continue
                    
            print(f"Import Summary:")
            print(f"- Successfully imported: {imported_count} properties")
            print(f"- Skipped due to errors: {skipped_count} rows")
            print(f"- Total rows processed: {imported_count + skipped_count}")
            
            if error_details:
                print(f"\nFirst 10 errors:")
                for error in error_details[:10]:
                    print(f"  {error}")
                
            return imported_count
            
    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
        return 0
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return 0

@app.post("/crm/create_user")
async def create_user(user: UserRegistration):
    if user.user_id in users:
        return {"error": "User ID already exists"}
    
    hash_password = hash_password(user.password)

    new_user = User(
        user_id = user.user_id,
        email = user.email,
        password_hash = hash_password
    )

    users[user.user_id] = new_user
    user_conversations[user.user_id] = ConversationalOpenAI()

    return {"message": "User created successfully", "user_id": user.user_id}

@app.post("/crm/login")
async def login_user(user: UserLogin):
    """Login user and return user details if successful"""
    user = auth_user(user.user_id, user.password)
    
    if not user:
        return {"error": "Invalid user ID or password"}
    
    # Return user details without password
    return {
        "user_id": user.user_id,
        "email": user.email,
        "message": "Login successful"
    }

@app.get("/properties")
async def get_properties():
    return properties

@app.post("/upload_docs")
async def upload_docs(csv_file_path: str):
    """Import properties from a CSV file"""
    count = import_properties_from_csv(csv_file_path)
    return {"message": f"Imported {count} properties from CSV", "total_properties": len(properties)}

@app.post("/debug/reload-ai")
async def reload_ai():
    """Reload the AI instance with fresh environment variables"""
    global ai
    try:
        # Force reload environment variables
        load_dotenv(override=True)
        # Create new AI instance
        ai = ConversationalOpenAI()
        new_key = os.getenv("OPENAI_API_KEY")
        return {
            "message": "AI instance reloaded successfully",
            "new_api_key_starts_with": new_key[:10] + "..." if new_key else "None",
            "key_length": len(new_key) if new_key else 0
        }
    except Exception as e:
        return {"error": f"Failed to reload AI: {str(e)}"}

# Property query and CSV generation functions
def query_properties(question: str):
    """Analyze properties based on user questions and return relevant data"""
    question_lower = question.lower()
    
    if not properties:
        return "No properties are currently loaded in the database."
    
    # Basic statistics
    if any(word in question_lower for word in ["how many", "total", "count"]):
        return f"There are {len(properties)} properties in the database."
    
    if any(word in question_lower for word in ["average rent", "avg rent", "mean rent"]):
        avg_rent = sum(p.annual_rent for p in properties) / len(properties)
        return f"The average annual rent is ${avg_rent:,.2f}"
    
    if any(word in question_lower for word in ["highest rent", "most expensive", "maximum rent"]):
        max_prop = max(properties, key=lambda p: p.annual_rent)
        return f"The highest rent is ${max_prop.annual_rent:,.2f} at {max_prop.property_address}, Floor {max_prop.floor}, Suite {max_prop.suite}"
    
    if any(word in question_lower for word in ["lowest rent", "cheapest", "minimum rent"]):
        min_prop = min(properties, key=lambda p: p.annual_rent)
        return f"The lowest rent is ${min_prop.annual_rent:,.2f} at {min_prop.property_address}, Floor {min_prop.floor}, Suite {min_prop.suite}"
    
    if any(word in question_lower for word in ["largest", "biggest", "maximum size"]):
        largest_prop = max(properties, key=lambda p: p.size_SF)
        return f"The largest property is {largest_prop.size_SF} SF at {largest_prop.property_address}, Floor {largest_prop.floor}, Suite {largest_prop.suite}"
    
    # Return general info if no specific query matched
    return f"I have access to {len(properties)} properties. You can ask about average rent, highest/lowest rent, largest property, or request a CSV export."

def generate_csv_from_properties():
    """Generate CSV content from current properties"""
    if not properties:
        return None, "No properties available to export"
    
    import io
    output = io.StringIO()
    
    # Write header
    output.write("unique_id,Property Address,Floor,Suite,Size (SF),Rent/SF/Year,Associate 1,BROKER Email ID,Associate 2,Associate 3,Associate 4,Annual Rent,Monthly Rent,GCI On 3 Years\n")
    
    # Write data
    for prop in properties:
        output.write(f"{prop.unique_id},{prop.property_address},{prop.floor},{prop.suite},{prop.size_SF},{prop.rent_SF_year},{prop.associate_1},{prop.broker_email_id},{prop.associate_2},{prop.associate_3},{prop.associate_4},{prop.annual_rent},{prop.monthly_rent},{prop.gci_on_3_years}\n")
    
    csv_content = output.getvalue()
    output.close()
    
    return csv_content, None

@app.get("/debug/api-key")
async def debug_api_key():
    """Debug endpoint to check API key status"""
    api_key = os.getenv("OPENAI_API_KEY")
    return {
        "api_key_exists": api_key is not None,
        "api_key_length": len(api_key) if api_key else 0,
        "api_key_starts_with": api_key[:10] + "..." if api_key and len(api_key) > 10 else "Invalid",
        "api_key_format_valid": api_key.startswith("sk-") if api_key else False,
        "env_file_exists": os.path.exists(".env"),
        "dotenv_loaded": True  # Since load_dotenv() was called
    }

@app.get("/test/property-query")
async def test_property_query(question: str = "how many properties"):
    """Test endpoint to check if property queries work"""
    result = query_properties(question)
    return {
        "question": question,
        "result": result,
        "properties_count": len(properties),
        "properties_loaded": len(properties) > 0
    }

@app.get("/export/csv")
async def export_properties_csv():
    """Export current properties as CSV"""
    csv_content, error = generate_csv_from_properties()
    
    if error:
        return {"error": error}
    
    from fastapi.responses import Response
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=properties_export.csv"}
    )

@app.get("/debug/csv-analysis")
async def analyze_csv(csv_file_path: str = "HackathonInternalKnowledgeBase.csv"):
    """Analyze CSV file to identify potential import issues"""
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            total_rows = 0
            issues = []
            sample_data = []
            column_names = reader.fieldnames
            
            for row_num, row in enumerate(reader, start=2):
                total_rows += 1
                
                # Check for common issues
                if not row.get('Property Address', '').strip():
                    issues.append(f"Row {row_num}: Missing Property Address")
                
                if not row.get('unique_id', '').strip():
                    issues.append(f"Row {row_num}: Missing unique_id")
                
                # Sample first 3 rows for inspection
                if len(sample_data) < 3:
                    sample_data.append({
                        "row_number": row_num,
                        "data": dict(row)
                    })
            
            return {
                "csv_file": csv_file_path,
                "total_rows_in_csv": total_rows,
                "column_names": column_names,
                "issues_found": len(issues),
                "first_10_issues": issues[:10],
                "sample_rows": sample_data,
                "properties_currently_loaded": len(properties)
            }
            
    except FileNotFoundError:
        return {"error": f"File not found: {csv_file_path}"}
    except Exception as e:
        return {"error": f"Error analyzing CSV: {str(e)}"}