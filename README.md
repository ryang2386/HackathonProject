# Hackathon Project

## Description

The purpose of this app is to create a Python-based RESTful API app that uses Large Language Models to retrieve, capture, and store information. Currently, this app can retrieve any information from the database that the user desires, and the app can create users, save, retrieve, and clear user chats, and upload CSV files to the app. Although the app isn't complete, great steps have been made to complete this app nearly.

## Table of Contents

* [Installation](#installation)
* [Usage](#usage)
* [License](#license)
* [Contribution](#contribution)
* [Testing](#testing)
* [Contact](#contact)

## Installation

First, many packages are being used in this app, and it may be required to install them before you can run the app.

**Install Packages**

   ```bash
   pip install -insert Python package here-
   ```
  
Or you can also install the packages using:

  ```bash
   python -m pip install -insert Python package here-
   ```

Before running the app, uvicorn must be installed. You can do so by typing this command in the terminal:

   ```bash
   python -m pip install uvicorn[standard]
   ```
   
After the packages are installed, you can run the program using this command:

   ```bash
   python -m uvicorn app:app --reload
   ```

## Usage

This app can be used by various companies that want quick access to information regarding properties that they manage. This app gives you the ability to create user IDs for each of your employees, and each employee would have the ability to upload CSV documents and ask the LLM specific information about the documents. Soon, the ability to retrieve, capture, and store information through chats will be available. Overall, this app can be a tool utilized by any company that want easier ways to parse through files and retrieve the information that they want easily.

Here's an example of a conversation with the LLM regarding the property information in the database:

<img width="1914" height="1065" alt="Screenshot 2025-07-13 175332" src="https://github.com/user-attachments/assets/e818cd51-d1ab-4961-9ec3-edb5acf02113" />

"Property data context" was not a part of the user's question in this example.

## License

None.

## Contribution

Contributions are welcome! To contribute, fork the repo and create a pull request. Once that is done, I will review it at my earliest convenience. I am fairly new to Python and still learning, so any recommendations would be greatly appreciated.

## Testing

Once the app is running, go to https://[localhost]/docs to use FastAPI to test the RESTful API. You'll be able to test each feature there.

## Contact

GitHub: https://github.com/ryang2386

LinkedIn: https://www.linkedin.com/in/ryangayle/

Email: gayler02@nyu.edu
