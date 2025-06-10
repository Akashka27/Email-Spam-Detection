import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load and prepare data
data = pd.read_csv("/Users/akashakash/Downloads/spam.csv")
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# Split data
mess_train, mess_test, cat_train, cat_test = train_test_split(
    data['Message'], data['Category'], test_size=0.2
)

# Vectorize and train
cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)
model = MultinomialNB()
model.fit(features, cat_train)

# Streamlit UI
st.header("SPAM DETECTION")
user_input = st.text_input("Enter Message Here", "Congratulations! You won a lottery")

if st.button('Validate'):
    # Make prediction
    input_vec = cv.transform([user_input])
    prediction = model.predict(input_vec)[0]
    
    # Display with color coding
    if prediction == "Spam":
        st.error(f"🚨 SPAM ALERT: {prediction}")
    else:
        st.success(f"✅ Not Spam: {prediction}")
    
    # Show raw message for debugging
    st.text(f"Message analyzed: {user_input}")