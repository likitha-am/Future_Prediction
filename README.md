# Future_Prediction
# Career Prediction ML App

This is a machine learning-powered web application that predicts a user's future career based on several factors like education, subjects of interest, tech-savviness, financial status, and more. The model is trained using XGBoost and SVM in an ensemble, and the interface is built with Streamlit.

---

## Project Files

- model_training.py         -> Trains the model and saves necessary objects (model, encoders, etc.)
- app.py                    -> Streamlit UI to take input and show career prediction
- xceldoc.csv               -> Cleaned input dataset
- trained_model.pkl         -> Trained ensemble model
- tfidf_vectorizer.pkl      -> TF-IDF vectorizer for text features
- scaler.pkl                -> StandardScaler for numerical features
- label_encoder.pkl         -> LabelEncoder for target classes
- ohe_objects.pkl           -> List of OneHotEncoders for categorical features
- requirements.txt          -> List of required Python libraries
- README.md                 -> This file

---

## How to Run

1. Clone this repository  
2. Install dependencies using `pip install -r requirements.txt`  
3. (Optional) Run `model_training.py` to train your model and generate `.pkl` files  
4. Launch the app with `streamlit run app.py`

---

## Input Features for Prediction

- Age  
- Highest Education Level  
- Preferred Subjects  
- Academic Performance (CGPA or %)  
- Preferred Work Environment  
- Risk-Taking Ability  
- Tech-Savviness  
- Financial Stability (1 to 10)

---

## Output

The model predicts a possible career path based on the inputs. Examples include:

- Software Engineer  
- Data Scientist  
- Entrepreneur  
- Teacher  
- Artist  
... and more

---

## Technologies Used

- Python  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Pandas / NumPy  
- TfidfVectorizer  
- OneHotEncoder / LabelEncoder / StandardScaler

---

## License

This project is free and open-source.
