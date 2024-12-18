# YouTube Category Performance Analyzer

This tool analyzes content categories on YouTube using AI categorization to track views and upload metrics. It identifies potential content opportunities by calculating views/upload scores for different categories.

## Content Gap App

Interactive Streamlit app with three features:
1. Generate video concepts, get AI categorization with top 3 similar categories, and assings content gap score
2. Assings content gap score for user-provided video concepts
3. Same for video titles

*example analysis*
![example analysis](./data/example_01.png)

### Setup

1. Clone the Repo
2. Create virtual environment
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create `.env` file and add openai key (for concept/title generation):
   ```
   OPENAI_API_KEY=your_key_here
   ```
5. Run the app:
   ```bash
   streamlit run app_content_gap.py
   ```

## Analysis Results

The analysis is based on a subsample of YouTube videos from our analytics assets catalog.

![Level 1 Categories Performance](./data/level1_categories.png)
*Views/Upload for Level 1 Categories*

![Challenges & Games Subcategories](./data/level2_categories.png)
*Views/Upload for Level 2 Categories under 'Challenges & Games'*

Data source: `data/category_performance_v2.xlsx`
