import streamlit as st
from joblib import load

# Main title and subheader
st.header("Toxic Comments Classification")
st.subheader("Enter Sentence")

# Function to show input fields
def show_page():
    comment_text = st.text_input("Enter the sentence:")
    return comment_text

sentence = show_page()

# Button to trigger prediction
if st.button("Predict"):
    if sentence:
        try:
            # Load the model
            text_clf_multiclass = load('tccc.pkl')

            # Make a prediction (including TF-IDF transformation)
            prediction = text_clf_multiclass.predict([sentence])

            # Define result categories
            result_categories = {
                0: "Non-Toxic",
                1: "Severely Toxic",
                2: "Toxic"
            }

            # Display the prediction result
            st.success(f"The prediction is: {result_categories.get(prediction[0], 'Unknown')}")
        except Exception as e:
            st.error(f"An error occurred: {e}")