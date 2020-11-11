# Fake News Classifier
## About
This repository contains efforts of a research project that I've been working on since December 2019. It started as a capstone project for the data science emphasis of my statistics BS degree from BYU. The research was so fascinating that I decided to continue working on it independently under Dr. Quinn Snell of the BYU CS dept.
## Files
### combined_model
Contains the training code for the pyTorch classification model.
### feature_embeddings
Contains the class for embedding features from a given web article. The current features are as follows:
- headline embeddings
For each article, the headline of said article is embedded as a 1x100 vector using an average word vector approach.
- article embeddings
