import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Scrape the free courses from Analytics Vidhya
url = "https://courses.analyticsvidhya.com/pages/all-free-courses"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

courses = []

# Extracting course title, image, and course link
for course_card in soup.find_all('header', class_='course-card__img-container'):
    img_tag = course_card.find('img', class_='course-card__img')
    
    if img_tag:
        title = img_tag.get('alt')
        image_url = img_tag.get('src')
        
        link_tag = course_card.find_previous('a')
        if link_tag:
            course_link = link_tag.get('href')
            if not course_link.startswith('http'):
                course_link = 'https://courses.analyticsvidhya.com' + course_link

            courses.append({
                'title': title,
                'image_url': image_url,
                'course_link': course_link
            })

# Step 2: Create DataFrame
df = pd.DataFrame(courses)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate embeddings using BERT
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Create embeddings for course titles
df['embedding'] = df['title'].apply(lambda x: get_bert_embedding(x))

# Function to perform search using BERT-based similarity
def search_courses(query):
    query_embedding = get_bert_embedding(query)
    course_embeddings = np.vstack(df['embedding'].values)
    
    # Compute cosine similarity between query embedding and course embeddings
    similarities = cosine_similarity(query_embedding, course_embeddings).flatten()
    
    # Add the similarity scores to the DataFrame
    df['score'] = similarities
    
    # Sort by similarity score in descending order and return top results
    top_results = df.sort_values(by='score', ascending=False).head(10)
    return top_results[['title', 'image_url', 'course_link', 'score']].to_dict(orient='records')

# Streamlit Interface
st.title("Analytics Vidhya Smart Course Search")
st.write("Find the most relevant courses from Analytics Vidhya based on your query.")

query = st.text_input("Enter your search query", placeholder="e.g., machine learning, data science, python")

if query:
    results = search_courses(query)
    if results:
        for item in results:
            course_title = item['title']
            course_image = item['image_url']
            course_link = item['course_link']
            relevance_score = round(item['score'] * 100, 2)
            
            st.image(course_image, width=300)
            st.markdown(f"### [{course_title}]({course_link})")
            st.write(f"Relevance: {relevance_score}%")
            st.markdown("---")
    else:
        st.write("No results found.")
