import re
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# Load the embedding model
embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# Load data
channel_metadata = pd.read_excel('data/channel_metadata.xlsx')
category_dict = pd.read_parquet('data/cluster_df.parquet')
cat_performance = pd.read_excel('data/category_performance_v2.xlsx')
cat_performance['upload_year'] = pd.to_datetime(cat_performance['upload_month']).dt.year
cat_performance = cat_performance.groupby(['category','category_index', 'upload_year'], as_index=False).agg({'num_videos': 'sum','views': 'sum'})
cat_performance['views_per_upload'] = cat_performance['views'] / cat_performance['num_videos']
cat_performance = cat_performance.sort_values(by=['upload_year', 'views_per_upload'], ascending=[False, False]).reset_index(drop=True)

# Calculate annual medians and ratios
annual_medians = cat_performance.groupby('upload_year')['views_per_upload'].median()
cat_performance['median_views_per_upload'] = cat_performance['upload_year'].map(annual_medians)
cat_performance['vpu_by_median'] = cat_performance['views_per_upload'] / cat_performance['median_views_per_upload']
cat_performance_2023 = cat_performance[cat_performance['upload_year'] == 2023].reset_index(drop=True)


common_words = {'subscribe','like','likes','comment','comments','video','videos','click','watch','let us know','share','channel','view','views','youtube','copyrights','please','watching','thanks','facebook','twitter','instagram','tiktok'}

def text_pre_process(text):
    processed_text = keep_first_two_paragraphs(text).lower()
    processed_text = remove_paragraphs_with_long_urls(processed_text)
    processed_text = remove_urls(processed_text)
    processed_text = remove_emails(processed_text)
    processed_text = remove_words_before_colon(processed_text)
    processed_text = remove_emojis(processed_text)
    processed_text = remove_hashtags(processed_text)
    processed_text = processed_text.replace('\n', ' ')
    processed_text = processed_text.replace('-', '').replace('_','').replace(',','').replace('!','').replace('|','')
    processed_text = re.sub(r'\s+', ' ', processed_text) # removing extra spaces
    processed_text = " ".join([word for word in processed_text.split() if word not in common_words])
    return processed_text

def remove_emails(text):
    # Regular expression pattern for matching email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # Replace found email addresses with an empty string
    return re.sub(email_pattern, '', text)

def remove_paragraphs_with_long_urls(text):
    # Regular expression pattern for matching URLs
    url_pattern = r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Function to determine if the paragraph should be kept
    def keep_paragraph(paragraph):
        urls = re.findall(url_pattern, paragraph)
        for url in urls:
            if len(url) >= 0.2 * len(paragraph):
                return False
        return True

    # Split text into paragraphs
    paragraphs = re.split(r'\n\n|\r\n\r\n', text)

    # Keep paragraphs that don't contain long URLs
    filtered_paragraphs = [paragraph for paragraph in paragraphs if keep_paragraph(paragraph)]

    return '\n\n'.join(filtered_paragraphs)

def keep_first_two_paragraphs(text):
    paragraphs = re.split(r'\n\n|\r\n\r\n', text)
    return '\n\n'.join(paragraphs[:2])

def remove_paragraphs_with_links_and_following(text):
    # Regular expression pattern for matching URLs
    url_pattern = r'https?://\S+|www\.\S+'
    
    # Split the text into paragraphs
    paragraphs = text.split('\n')

    # Initialize an empty list to hold the filtered paragraphs
    filtered_paragraphs = []

    # Iterate over each paragraph
    for para in paragraphs:
        # Check if the paragraph contains a URL
        if re.search(url_pattern, para):
            # If a URL is found, break out of the loop
            break
        # Otherwise, add the paragraph to the filtered list
        filtered_paragraphs.append(para)

    # Rejoin the remaining paragraphs
    return '\n'.join(filtered_paragraphs)

def remove_paragraphs_with_links(text):
    # Regular expression pattern for matching URLs
    url_pattern = r'https?://\S+|www\.\S+'
    
    # Split the text into paragraphs
    paragraphs = text.split('\n')

    # Filter out paragraphs that contain URLs
    filtered_paragraphs = [para for para in paragraphs if not re.search(url_pattern, para)]

    # Rejoin the remaining paragraphs
    return '\n'.join(filtered_paragraphs)

def remove_words_before_colon(text):
    # Regular expression pattern to match any text up to and including a colon
    pattern = r'\b\w+\s*:'

    # Split the text into lines or sentences
    lines = text.split('\n')

    # Remove the matched pattern from each line
    cleaned_lines = [re.sub(pattern, '', line).strip() for line in lines]

    # Rejoin the cleaned lines
    return '\n'.join(cleaned_lines)

def remove_hashtags(text):
    # Regular expression pattern to match any text up to and including a colon
    pattern = r'#\w+\b'

    # Split the text into lines or sentences
    lines = text.split('\n')

    # Remove the matched pattern from each line
    cleaned_lines = [re.sub(pattern, '', line).strip() for line in lines]

    # Rejoin the cleaned lines
    return '\n'.join(cleaned_lines)

def remove_urls(text):
    # Regular expression pattern to match any text up to and including a colon
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Split the text into lines or sentences
    lines = text.split('\n')

    # Remove the matched pattern from each line
    cleaned_lines = [re.sub(pattern, '', line).strip() for line in lines]

    # Rejoin the cleaned lines
    return '\n'.join(cleaned_lines)


def remove_emojis(text):
    # Regex pattern to match emojis
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251" 
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# Helper functions
def fetch_creator_metadata(creator):
    creator_metadata = channel_metadata[channel_metadata['author'] == creator]
    if creator_metadata.empty:
        raise ValueError("No metadata found for the specified creator.")
    return creator_metadata

def preprocess_text(text):
    return text_pre_process(text) if text else ""

def generate_embedding(text):
    return embedding_model.encode([text]) if text else None

def prepare_embeddings(creator_metadata, title):
    channel_description = preprocess_text(creator_metadata['description'].iloc[0] if not creator_metadata['description'].isna().all() else "")
    channel_keywords = preprocess_text(creator_metadata['keywords'].iloc[0] if not creator_metadata['keywords'].isna().all() else "")
    
    title_embed = generate_embedding(title)
    channel_description_embed = generate_embedding(channel_description)
    channel_tags_embed = generate_embedding(channel_keywords)

    embeddings = [title_embed, channel_description_embed, channel_tags_embed]
    weights = [10, 3, 3]

    # Filter valid embeddings
    valid_embeddings = [(emb, weight) for emb, weight in zip(embeddings, weights) if emb is not None and emb.size > 0]

    if not valid_embeddings:
        raise ValueError("No valid embeddings found for any component.")

    emb_list, weight_list = zip(*valid_embeddings)
    return emb_list, weight_list

def normalize_weights(weights):
    normalized_weights = (weights - 0.7) / (1 - 0.7)
    normalized_weights = normalized_weights / normalized_weights.sum()
    return normalized_weights

def compute_category_similarity(creator, title):
    creator_metadata = fetch_creator_metadata(creator)
    emb_list, weight_list = prepare_embeddings(creator_metadata, title)

    emb_array = np.array(emb_list).transpose(1, 2, 0)
    emb_tensor = torch.from_numpy(emb_array).type(torch.DoubleTensor)
    weight_tensor = torch.tensor(weight_list).type(torch.DoubleTensor).unsqueeze(1)

    emb_tensor_weighted = torch.matmul(emb_tensor, weight_tensor).squeeze()
    emb_tensor_weighted_normalized = F.normalize(emb_tensor_weighted.reshape(1, -1) if len(emb_tensor_weighted.shape) < 2 else emb_tensor_weighted, p=2, dim=1)

    centroids_tensor = torch.tensor(category_dict['centroids'].tolist(), dtype=torch.double)
    cosine = F.cosine_similarity(emb_tensor_weighted_normalized.unsqueeze(1), centroids_tensor.unsqueeze(0), dim=2)

    cluster_dic = []
    for cos_sim in cosine:
        top_3_indices = torch.topk(cos_sim, k=3).indices
        top_3_similarities = cos_sim[top_3_indices].tolist()
        cluster_dic.append(top_3_similarities + top_3_indices.tolist() + [category_dict['category_levels'][i] for i in top_3_indices.tolist()] )

    df_ = pd.DataFrame(cluster_dic, columns=['similarity_1', 'similarity_2', 'similarity_3', 'idx_1', 'idx_2', 'idx_3', 'category_1', 'category_2', 'category_3'])
    # Assign scores based on category performance
    for i in range(1, 4):
        try:
            matched_score = cat_performance_2023[cat_performance_2023['category_index'] == df_[f'idx_{i}'][0]]['vpu_by_median'].iloc[0]
            df_[f'cat{i}_gap_score'] = matched_score
        except IndexError:
            df_[f'cat{i}_gap_score'] = np.nan 
    similarity_weights = normalize_weights(np.array([df_[f'similarity_{i}'] for i in range(1, 4)])) # Normalize similarity weights
 
    # Filter valid weights and scores
    cat_gap_scores = np.array([df_[f'cat{i}_gap_score'] for i in range(1, 4)])
    valid_mask = ~np.isnan(cat_gap_scores)

    if np.all(~valid_mask):
        df_['total_score'] = np.nan
    else:
        valid_weights = similarity_weights[valid_mask]
        valid_scores = cat_gap_scores[valid_mask]
        
        valid_weights = valid_weights / valid_weights.sum()
        df_['total_score'] = np.dot(valid_weights, valid_scores)
    
    return df_