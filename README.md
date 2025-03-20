# Multilingual Text Classifier

## Project Overview  
This web application detects the language of input text using machine learning. Built with **Flask** and **scikit-learn**, it employs a **Multinomial Naïve Bayes classifier** trained on text samples across multiple languages to identify the language of any provided text snippet.

## Features  
✅ Real-time language detection for text input  
✅ Supports multiple languages based on the training dataset  
✅ Simple web interface for easy interaction  
✅ Fast processing with an optimized machine learning model  

## Technical Details  
The application uses **Natural Language Processing (NLP)** techniques to analyze text patterns:  
- **Text Preprocessing**: Removes special characters and normalizes data  
- **Feature Extraction**: Converts text into numerical vectors using `CountVectorizer`  
- **Classification**: Predicts the language using **Multinomial Naïve Bayes**  
- **Web Framework**: Flask is used to serve the application  

## Installation  

### **Prerequisites**  
- Python **3.7+**  
- pip package manager  

### **Required Packages**  
Install the dependencies using the following command:  
```bash
pip install -r requirements.txt
```
Or install them manually:  
```bash
pip install flask pandas numpy scikit-learn scipy
```

## How It Works  
1️⃣ **Data Preprocessing**: Cleans text by removing special characters and converting to lowercase.  
2️⃣ **Feature Extraction**: Converts text to feature vectors using `CountVectorizer`.  
3️⃣ **Classification**: Uses **Multinomial Naïve Bayes** to predict the language.  
4️⃣ **Response**: Returns the detected language to the user.  

## Project Structure  
```
language-detection/
├── app.py                  # Main application file
├── templates/              # HTML templates
│   └── index.html          # User interface
├── Language_Detection.csv  # Dataset for training
├── requirements.txt        # Required packages
└── README.md               # Project documentation
```

## Future Improvements  
🚀 **Add support for more languages**  
📊 **Implement confidence scores for predictions**  
🌐 **Create a REST API endpoint for programmatic access**  
🧠 **Enhance accuracy using more advanced NLP techniques**  

---

