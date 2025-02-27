import streamlit as st
import nltk
from nltk import NaiveBayesClassifier
from nltk.classify import apply_features
from joblib import load

# Download NLTK resources if not already downloaded
nltk.download('names')

# Function to extract features from a name
def extract_gender_features(name):
    name = name.lower()
    features = {
        "suffix": name[-1:],
        "suffix2": name[-2:] if len(name) > 1 else name[0],
        "suffix3": name[-3:] if len(name) > 2 else name[0],
        "suffix4": name[-4:] if len(name) > 3 else name[0],
        "suffix5": name[-5:] if len(name) > 4 else name[0],
        "suffix6": name[-6:] if len(name) > 5 else name[0],
        "prefix": name[:1],
        "prefix2": name[:2] if len(name) > 1 else name[0],
        "prefix3": name[:3] if len(name) > 2 else name[0],
        "prefix4": name[:4] if len(name) > 3 else name[0],
        "prefix5": name[:5] if len(name) > 4 else name[0]
    }
    return features

# Load the trained Naive Bayes classifier
bayes = load('gender_prediction.joblib')

# Streamlit app
def main():
    st.title('Gender Prediction App')
    st.write('Enter a name to predict its gender.')

    # Input for name
    input_name = st.text_input('Name:')
    
    if st.button('Predict'):
        if input_name.strip() != '':
            # Extract features for the input name
            features = extract_gender_features(input_name)
            
            # Predict using the trained classifier
            predicted_gender = bayes.classify(features)
            
            # Display prediction
            st.success(f'The predicted gender for "{input_name}" is: {predicted_gender}')
        else:
            st.warning('Please enter a name.')

if __name__ == '__main__':
    main()
