import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def get_headers():
    return {
        'authority': 'www.amazon.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'accept-language': 'en-US,en;q=0.9,bn;q=0.8',
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
    }


def get_reviews_url():
    return 'https://www.amazon.com/Fitbit-Smartwatch-Readiness-Exercise-Tracking/product-reviews/B0B4MWCFV4/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'


def reviewsHtml(url, len_page):
    headers = get_headers()
    soups = []
    for page_no in range(1, len_page + 1):
        params = {
            'ie': 'UTF8',
            'reviewerType': 'all_reviews',
            'filterByStar': 'critical',
            'pageNumber': page_no,
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'lxml')
        soups.append(soup)
    return soups


def get_reviews_data(html_data):
    data_dicts = []
    boxes = html_data.select('div[data-hook="review"]')
    for box in boxes:
        try:
            name = box.select_one('[class="a-profile-name"]').text.strip()
        except Exception as e:
            name = 'N/A'
        try:
            stars = box.select_one('[data-hook="review-star-rating"]').text.strip().split(' out')[0]
        except Exception as e:
            stars = 'N/A'
        try:
            title = box.select_one('[data-hook="review-title"]').text.strip()
        except Exception as e:
            title = 'N/A'
        try:
            datetime_str = box.select_one('[data-hook="review-date"]').text.strip().split(' on ')[-1]
            date = datetime.strptime(datetime_str, '%B %d, %Y').strftime("%d/%m/%Y")
        except Exception as e:
            date = 'N/A'
        try:
            description = box.select_one('[data-hook="review-body"]').text.strip()
        except Exception as e:
            description = 'N/A'
        data_dict = {
            'Name': name,
            'Stars': stars,
            'Title': title,
            'Date': date,
            'Description': description
        }
        data_dicts.append(data_dict)
    return data_dicts


def process_data(html_datas, len_page):
    reviews = []
    for html_data in html_datas:
        review = get_reviews_data(html_data)
        reviews += review
    df_reviews = pd.DataFrame(reviews)
    return df_reviews


def clean_data(df_reviews):
    df_reviews['Description'] = df_reviews['Description'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    df_reviews['Description'] = df_reviews['Description'].apply(lambda x: x.lower())
    stop_words = set(stopwords.words('english'))
    df_reviews['Description'] = df_reviews['Description'].apply(
        lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))
    lemmatizer = WordNetLemmatizer()
    df_reviews['Description'] = df_reviews['Description'].apply(
        lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))
    df_reviews.to_csv('cleaned_reviews.csv', index=False)
    print("Data processing and cleaning completed.")
    return df_reviews


def analyze_sentiment(description):
    analysis = TextBlob(description)
    sentiment = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    confidence = abs(sentiment) + (1 - subjectivity) * 100

    if sentiment > 0:
        return 'Positive', confidence
    elif sentiment < 0:
        return 'Negative', confidence
    else:
        return 'Neutral', confidence


def train_data(df_reviews):
    df_reviews[['Sentiment', 'Confidence']] = df_reviews['Description'].apply(analyze_sentiment).apply(pd.Series)
    return df_reviews[['Description', 'Sentiment', 'Confidence']]


def visualize_data(df_reviews):
    st.subheader("Visualized Data:")

    st.subheader("Sentiment Distribution:")
    sentiment_counts = df_reviews['Sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    st.subheader("Word Cloud:")
    all_words = ' '.join(df_reviews['Description'])
    generate_wordcloud_st(all_words)

    st.subheader("Pie Chart:")
    visualize_pie_chart(df_reviews)

    st.subheader("Histogram:")
    visualize_histogram(df_reviews)


def visualize_pie_chart(df_reviews):
    sentiment_counts = df_reviews['Sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis'),
           startangle=90)
    ax.axis('equal')
    st.pyplot(fig)


def visualize_histogram(df_reviews):
    plt.figure(figsize=(10, 6))
    sns.histplot(df_reviews['Confidence'], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of Sentiment Confidence Scores')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    st.pyplot()


def generate_wordcloud(sentiment, df_reviews):
    words = ' '.join(df_reviews[df_reviews['Sentiment'] == sentiment]['Description'])
    if not words:
        print(f"No reviews for {sentiment} sentiment.")
        return

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment} Reviews')
    plt.show()


def analyze_sentiment_st(description):
    analysis = TextBlob(description)
    sentiment = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    confidence = abs(sentiment) + (1 - subjectivity) * 100

    if sentiment > 0:
        return 'Positive', confidence
    elif sentiment < 0:
        return 'Negative', confidence
    else:
        return 'Neutral', confidence


def generate_wordcloud_st(words):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    st.pyplot(fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)


def import_data(file_path):
    df = pd.read_csv(file_path)
    return df


def clean_and_store_data(df, csv_filename='cleaned_reviews.csv'):
    # Clean data
    df['Description'] = df['Description'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    df['Description'] = df['Description'].apply(lambda x: x.lower())
    stop_words = set(stopwords.words('english'))
    df['Description'] = df['Description'].apply(
        lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))
    lemmatizer = WordNetLemmatizer()
    df['Description'] = df['Description'].apply(
        lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

    # Store cleaned data in a new CSV
    cleaned_csv_path = csv_filename
    df.to_csv(cleaned_csv_path, index=False)

    return cleaned_csv_path


def main():
    st.title("Amazon Reviews Sentiment Analysis App")

    option = st.sidebar.selectbox("Choose an option", ["Write Review", "Enter Amazon URL", "Import CSV"])

    if option == "Import CSV":
        st.header("Import CSV for Analysis")

        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            df[['Sentiment', 'Confidence']] = df['Description'].apply(analyze_sentiment_st).apply(pd.Series)

            st.subheader("Data Preview:")
            st.write(df.head())

            st.subheader("Visualized Data:")

            st.subheader("Sentiment Distribution:")
            sentiment_counts = df['Sentiment'].value_counts()
            st.bar_chart(sentiment_counts)

            st.subheader("Word Cloud:")
            all_words = ' '.join(df['Description'])
            generate_wordcloud_st(all_words)

            st.subheader("Pie Chart:")
            visualize_pie_chart(df)

            st.subheader("Histogram:")
            visualize_histogram(df)

    elif option == "Write Review":
        st.header("Write Review for Analysis")

        user_input = st.text_area("Enter your review:")

        if st.button("Analyze"):
            if user_input:
                result, confidence = analyze_sentiment_st(user_input)
                st.subheader("Sentiment Analysis Result:")
                st.write(f"Sentiment: {result}")
                st.write(f"Confidence Score: {confidence}")

            else:
                st.warning("Please enter a review for analysis.")

    elif option == "Enter Amazon URL":
        st.header("Enter Your Favourite Amazon product's URL")

        URL_input = st.text_input("Enter Valid Amazon URL:")

        page_len = st.slider("Select the number of pages to scrape", min_value=1, max_value=10, value=1)

        if st.button("Analyze"):
            if URL_input:
                html_datas = reviewsHtml(URL_input, page_len)
                df_reviews = process_data(html_datas, page_len)
                df_reviews = clean_data(df_reviews)
                cleaned_csv_path = clean_and_store_data(df_reviews)

                df_cleaned = import_data(cleaned_csv_path)
                df_cleaned[['Sentiment', 'Confidence']] = df_cleaned['Description'].apply(analyze_sentiment_st).apply(
                    pd.Series)

                st.subheader("Data Preview after Cleaning:")

                st.write(df_cleaned.head())

                visualize_data(df_cleaned)

            else:
                st.warning("Please enter a URL first!")


if __name__ == "__main__":
    main()
