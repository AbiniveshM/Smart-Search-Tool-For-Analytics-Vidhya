# Smart-Search-Tool-For-Analytics-Vidhya-Courses
# Goal
To create a smart search tool that enables users to find relevant free courses on Analytics Vidhya’s platform quickly.

# Project Approach
# Data Collection
I began by scraping the free courses' titles and relevant metadata, such as course links and images, from Analytics Vidhya’s platform using BeautifulSoup.

# Model Selection
Originally, I used the Groq API for generating embeddings and conducting searches. However, I found the results less suitable, leading me to switch to a more refined solution using BERT (Bidirectional Encoder Representations from Transformers). I leveraged a pre-trained BERT model (bert-base-uncased from Hugging Face) for generating embeddings.

# Relevance Matching
To match user queries with relevant courses, I calculated cosine similarity between the user’s query embedding and the course title embeddings. This similarity score enables ranking courses based on relevance, ensuring the most suitable courses are shown first.

# Interface
The application uses both Streamlit and Shiny for flexible, user-friendly interfaces. These interfaces display course details dynamically, including title, image, link, and relevance score.Finally I can able to conclude that Shiny is more faster in retrieving the results and display those in more interactive way than StreamLit.

# Deployment on Hugging Face Spaces
I deployed the tool on Hugging Face Spaces, providing an accessible, visually appealing interface for public use, enhanced with custom CSS for style and responsiveness.

BERT model : google-bert/bert-base-uncased
