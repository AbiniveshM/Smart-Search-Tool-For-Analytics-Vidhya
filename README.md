# Smart Search Tool for Free Courses on Analytics Vidhya
This project is a smart search tool designed to help users find the most relevant free courses on Analytics Vidhya’s platform based on their search queries. It leverages NLP techniques to match user queries with course titles, using a BERT (Bidirectional Encoder Representations from Transformers) model for high-quality search relevance.

# Project Overview
# Objective
To develop a smart search tool that improves the course discovery experience for users, allowing them to find the most relevant courses based on their input.

# Key Features
Data Collection: 
Course titles, links, and images are scraped from Analytics Vidhya's free courses page.
Embedding Generation: 
Using the pre-trained bert-base-uncased model to generate embeddings for course titles and search queries.
Relevance Ranking: 
Computes cosine similarity between query and course title embeddings, ranking the courses by relevance.
Real-Time Autocomplete: 
Autocomplete suggestions based on course titles for an improved user experience.
Interactive Interface: 
Developed with Streamlit, offering a clean, user-friendly interface.

# Implementation Details
Data Collection: 
Using BeautifulSoup to scrape course data from Analytics Vidhya’s platform.
Model and Similarity Scoring:
The bert-base-uncased model from Hugging Face was used to generate embeddings for course titles and search queries.
Cosine similarity between the query embedding and course title embeddings helps rank courses by relevance.
Streamlit Interface:
A Streamlit application presents the course titles, images, and relevance scores dynamically in an easy-to-use layout.
Deployment
The project has been deployed using Hugging Face Spaces for public access and testing.

# Setup and Usage
Prerequisites
Python 3.8+
Streamlit
Transformers library from Hugging Face
BeautifulSoup for web scraping
