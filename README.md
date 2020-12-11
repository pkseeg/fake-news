# Fake News Classifier
## About
This repository contains efforts of a research project that I've been working on since December 2019. It started as a capstone project for the data science emphasis of my statistics BS degree from BYU. The research was so fascinating that I decided to continue working on it independently under Dr. Quinn Snell of the BYU CS dept.
## Files
### combined_model
Contains the training code for the pyTorch classification model.
### feature_embeddings
Contains the class for embedding features from a given web article. The current features are as follows:
- headline embeddings
The headline of each article is embedded as a 1x100 vector using an average word vector approach.
- article embeddings
The text body of each article is embedded as a 1x100 vector using a doc2vec model approach, (see doc2vec_model)
- URL bigram model feature extraction
Each article url is fed through a bigram model that was trained on training data. This model measures the entropy and perplexity of both the clean (i.e. domain and subdomains only) and full URL strings for each article.
- URL edit distance
Sometimes fake news sites attempt to mimic real news sites by almost copying their site domain - this measure the minimum edit distance from a news article URL (cleaned version) to any of the top 15 news sites URLs.
### features
This is a .py file that contains the feature_embeddings class that was built in feature_embeddings.
### model_inference, scikit_learn_inference
These files contain inference steps for the pyTorch model and scikit learn models. The scikit_learn notebook also utilizes shap to show that the most important features come from the article embeddings and some of the headline embeddings.
### doc2vec_model
This file contains the training code for the article body doc2vec model. It uses the training dataset to create tagged documents, where each tag is the snopes fact rating response, allowing for additional semantic meaning when creating document vectors (see feature_embeddings).
### create_text_files
This file simply creates text files for each of the different snopes fact ratings. These files were created as a precursor to generating news articles using a leakGAN approach (still working on this).