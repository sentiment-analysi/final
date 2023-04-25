import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

nltk.download('stopwords')

# Load the trained model and preprocessing objects
classifier = load_model('trained_model.h5')
cv = pickle.load(open('count-Vectorizer.pkl','rb'))
sc = pickle.load(open('Standard-Scaler.pkl','rb'))

# Function to perform sentiment analysis
def predict_sentiment(input_review):
    input_review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=input_review)
    input_review = input_review.lower()
    input_review_words = input_review.split()
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    input_review_words = [word for word in input_review_words if not word in stop_words]
    ps = PorterStemmer()
    input_review = [ps.stem(word) for word in input_review_words]
    input_review = ' '.join(input_review)
    input_X = cv.transform([input_review]).toarray()
    input_X = sc.transform(input_X)
    pred = classifier.predict(input_X)
    pred = (pred > 0.5)
    if pred[0][0]:
        return "Positive review"
    else:
        return "Negative review"

def predict_sentiment1(df, column_name):
    input_reviews = df[column_name].tolist()
    input_reviews = [re.sub(pattern='[^a-zA-Z]', repl=' ', string=review) for review in input_reviews]
    input_reviews = [review.lower() for review in input_reviews]
    input_reviews = [review.split() for review in input_reviews]
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    input_reviews = [[word for word in review_words if not word in stop_words] for review_words in input_reviews]
    ps = PorterStemmer()
    input_reviews = [[ps.stem(word) for word in review_words] for review_words in input_reviews]
    input_reviews = [' '.join(review_words) for review_words in input_reviews]
    input_X = cv.transform(input_reviews).toarray()
    input_X = sc.transform(input_X)
    pred = classifier.predict(input_X)
    pred = (pred > 0.5)
    sentiment = ['Positive review' if p else 'Negative review' for p in pred]
    return sentiment
# Function to show the analytics in a separate tab
def show_analytics(df, column_name):
    # Apply sentiment analysis to specified column
    sentiments = predict_sentiment(df, column_name)
    
    # Get the count of reviews and positive/negative reviews
    total_reviews = len(sentiments)
    positive_reviews = sentiments.count('Positive review')
    negative_reviews = sentiments.count('Negative review')
    
    # Print the count of reviews and positive/negative reviews
    st.write(f"Total number of reviews: {total_reviews}")
    st.write(f"Number of positive reviews: {positive_reviews}")
    st.write(f"Number of negative reviews: {negative_reviews}")
    
    # Plot the sentiment analysis results using matplotlib
    fig, ax = plt.subplots()
    ax.bar(['Positive', 'Negative'], [positive_reviews, negative_reviews], color=['blue', 'orange'])
    ax.set_title('Sentiment Analysis Results')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    
# Main function to run the app
def run_sentiment_app():
    st.title('Sentiment Analysis App')

    # Add a menu with two options
    menu = ['Home', 'Analytics']
    choice = st.sidebar.selectbox('Select page', menu)

    # Show the home page if the user selects the 'Home' option
    if choice == 'Home':

        st.subheader('Answer the following questions')

        # Get the user inputs
        review1 = st.text_input('How was the course experience?')
        review2 = st.text_input('Tell us about the instructor?')
        review3 = st.text_input('Was the material provided useful?')

        # Perform sentiment analysis and show the results
        if st.button('Predict'):
            result1 = predict_sentiment(review1)
            result2 = predict_sentiment(review2)
            result3 = predict_sentiment(review3)
            st.success(f"Course experience: {result1}")
            st.success(f"Instructor: {result2}")
            st.success(f"Material: {result3}")

            # Show analytics using a bar chart
            results = {'Course experience': result1, 'Instructor': result2, 'Useful material': result3}
            df = pd.DataFrame({'Reviews': list(results.keys()), 'Sentiment': list(results.values())})
            df_counts = df['Sentiment'].value_counts()
            fig, ax = plt.subplots()
            ax.bar(df_counts.index, df_counts.values, color=['blue', 'yellow'])
            ax.set_title('Sentiment Analysis Results')
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Count')
            st.pyplot(fig)
            
            #counting positive and negative reviews
            positive_count = (df_counts['Positive review'] if 'Positive review' in df_counts else 0)
            negative_count = (df_counts['Negative review'] if 'Negative review' in df_counts else 0)
            st.write(f"Number of positive reviews: {positive_count}")
            st.write(f"Number of negative reviews: {negative_count}")

    # Show the analytics page if the user selects the 'Analytics' option
    elif choice == 'Analytics':
        st.subheader('Upload an excel file to perform sentiment analysis')
        file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
        if file is not None:
            df = pd.read_excel(file)
            column_name = st.selectbox('Select column to analyze:', df.columns)
            st.write(df)

            # Show analytics in a separate tab on click of a button
            if st.button('Show Analytics'):
                show_analytics(df, column_name)

# Run the app
if __name__=='__main__':
    run_sentiment_app()
