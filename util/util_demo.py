import concurrent.futures
import json
import numpy as np
import pandas as pd
import re
import streamlit as st
import util.util_cats as util_cats
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
openai_model="gpt-4"


overperformers = pd.read_csv('data/overperformers.csv',header=None)

def read_json(args):
    if type(args) == dict:
        channel_name = args['channel_name']
        timestamp = args['timestamp']
        path = f'output/{channel_name}/temp-{timestamp}.json'
    elif type(args) == str:
        path = args
        print(f"{path=}")
    if os.path.isfile(path):
        with open(path,encoding='utf-8') as f:
            return json.load(f)
    else:
        return None
    
# Helper to format titles list
def format_titles(titles):
    return ", ".join(f'"{title}"' for title in titles)

def titles_system_prompt(channel_name,channel_bio,inspired_by,titles_of_my_overperformers,titles_of_related_overperformers):

    if inspired_by == 'spotterBlend':
        inspired_section = (
            f"The new titles for {channel_name}'s channel must be inspired by the common themes and topics in two different set of video titles for YouTube videos. "
            f"The first set of video titles were already made by {channel_name} and overperformed for them in the past. "
            f"Use these titles as an example of the SENTENCE STRUCTURE THAT YOU MUST USE in your suggestions: [{format_titles(titles_of_my_overperformers)}] "
            f"Identify the most common words and USE THOSE COMMON WORDS IN ALL of your suggestions. Summarize the common topics and themes that may have made these videos successful but never replicate these ideas exactly. "
            f"The second set of video titles are not made by {channel_name} but overperformed with their audience. These videos are: [{format_titles(titles_of_related_overperformers)}] "
            f"Use the themes, topics, ideas, and common motifs contained in the titles above as inspiration for new titles that could work for {channel_name} and their channel. "
            # f"Pitch {channel_name} ideas that are bigger, bolder, faster-paced, more grandiose, more specific, more spectacular, and more unprecedented than any video they have ever made before. "
        )
    elif inspired_by == 'myOverPerformers':
        inspired_section = (
            f"For your reference, here are titles for some of the best performing videos {channel_name} has made in the past. Use these titles as an example of the SENTENCE STRUCTURE THAT YOU MUST USE in your suggestions: [{format_titles(titles_of_my_overperformers)}] "
            f"Identify the most common words and USE THOSE COMMON WORDS IN ALL of your suggestions. Take these titles as initial inspiration and put them on steroids. "
            f"Pitch {channel_name} ideas that are bigger, bolder, faster-paced, more grandiose, more specific, more spectacular, and more unprecedented than any video they have ever made before. "
    )
    elif inspired_by == 'relatedOverPerformers':
        inspired_section = (
            f"The new ideas must be inspired by the following set of video titles that were not made by {channel_name} but overperformed with their audience. " 
            f"These videos are: [{format_titles(titles_of_my_overperformers)}] "
            f"Use the themes, topics, ideas, and common motifs contained in the titles above as inspiration for new titles that could work for {channel_name} and their channel. "
        )
    else:
        inspired_section = (
            f"For your reference, here are titles for some of the best performing videos {channel_name} has made in the past. "
            f"Use these titles as an example of the SENTENCE STRUCTURE THAT YOU MUST USE in your suggestions: [{format_titles(titles_of_my_overperformers)}] "
            f"Identify the most common words and USE THOSE COMMON WORDS IN ALL of your suggestion. "
            f"Take these titles as initial inspiration and put them on steroids. "
            f"Pitch {channel_name} ideas that are bigger, bolder, faster-paced, more grandiose, more specific, more spectacular, and more unprecedented than any video they have ever made before. "
        )

    # Constructing the full template
    prompt = f"""
    You are a world famous YouTube title expert. You know what makes a great video title that immediately grabs viewers' attention and causes them to click on the video and will optimize all of your recommendations for high "clickability".
    You know that great YouTube video titles are intriguing, make the viewer stop scrolling and click, and NO MORE THAN 8 WORDS so that they can be read on mobile.
    Based on all of your research, you have developed "YouTube title strategies" that states that the best YouTube video titles:
    - create curiosity in the viewer
    - spark a sense of desire
    - trigger fear or a sense of negativity in the viewer
    - contain something unexpected
    - are clear, concise, and NO MORE THAN 8 WORDS
    - use keywords strategically
    - appeal to the viewer's emotions
    - are explanatory
    - are authentic
    - are relevant
    - are unique
    - aren't too long
    - include a call to action
    - DO NOT CONTAIN COLONS EVER!
    You have recently been hired by {channel_name} to pitch new video titles. {channel_bio}
    {inspired_section}
    Your Response should be a json object, with a single parameter named "elements". 
    "elements" should be a json array with multiple objects that have the fields: 'result' containing a generated concept, 
    'type' is always "title", and 'id' containing the index of the element. 
    Ensure that the array contains 6 objects.
    """
    prompt += '''For example, your response should look something like: { \""elements\"": [{ \""id\"": 1, \""type\"": \""title\"", \""result\"": \""story element 1\"" }, { \""id\"": 2, \""type\"": \""title\"", \""result\"": \""story element 2\"" }... { \""id\"": 6, \""type\"": \""title\"", \""result\"": \""story element 6\"" }]}'''

    return prompt

def titles_user_prompt(channel_name,topic=None):
    prompt = f"Generate 6 potential titles for {channel_name}'s newest YouTube video. "
    if topic != None:
        prompt += f"Each title must be uniquely inspired by the concept of '{topic}' and must be 8 words or less. "
    return prompt

count = 6
def titles_my_overperformers_system_prompt(channel_name,channel_bio, my_overperformers_titles,loaded_data):
    prompt = f'''
You are a world famous YouTube title expert. You know what makes a great video title that immediately grabs viewers' attention and causes them to click on the video and will optimize all of your recommendations for high \""\""clickability\""\"".
You know that great YouTube video titles are intriguing, make the viewer stop scrolling and click, and NO MORE THAN 8 WORDS so that they can be read on mobile.
Based on all of your research, you have developed \""\""YouTube title strategies\""\"" that states that the best YouTube video titles:
- create curiosity in the viewer
- spark a sense of desire
- trigger fear or a sense of negativity in the viewer
- contain something unexpected
- are clear, concise, and NO MORE THAN 8 WORDS
- use keywords strategically
- appeal to the viewer's emotions
- are explanatory
- are authentic
- are relevant
- are unique
- aren't too long
- include a call to action
- DO NOT CONTAIN COLONS EVER!
You have recently been hired by {channel_name} to pitch new video titles. {channel_bio}
For your reference, here are titles for some of the best performing videos {channel_name} has made in the past. Use these titles as an example of the SENTENCE STRUCTURE THAT YOU MUST USE in your suggestions:
\""\""\""
{my_overperformers_titles}
\""\""\""
Identify the most common words and USE THOSE COMMON WORDS IN ALL of your suggestions. Take these titles as initial inspiration and put them on steroids.
Pitch {channel_name} ideas that are bigger, bolder, faster-paced, more grandiose, more specific, more spectacular, and more unprecedented than any video they have ever made before.
Your Response should be a json object, with a single parameter named \""elements\"". 
\""elements\"" should be a json array with multiple objects that have the fields: 'result' containing a generated concept, 
'type' is always \""title\"", and 'id' containing the index of the element. 
Ensure that the array contains {count} objects.
'''
    prompt += '''\nFor example, your response should look something like: { \""elements\"": [{ \""id\"": 1, \""type\"": \""title\"", \""result\"": \""story element 1\"" }, { \""id\"": 2, \""type\"": \""title\"", \""result\"": \""story element 2\"" }... { \""id\"": 6, \""type\"": \""title\"", \""result\"": \""story element 6\""'''
    return prompt

# Avoid ideas that are infeasible or impossible to implement for this YouTube creator.

def titles_my_overperformers_user_prompt(channel_name):
    return f"Generate {count} potential titles for {channel_name}'s newest YouTube video. "

def get_bio(channel_name):
    with open(f"data/channels/{channel_name}.json", encoding='utf-8') as f:
        channel_info = json.load(f)
    return channel_info['bio']



def concepts_system_prompt(channel_name,channel_bio,inspired_by,titles_of_my_overperformers,titles_of_related_overperformers):
    if inspired_by == 'spotterBlend':
        inspired_section = (
            f"The new ideas for {channel_name}'s channel must be inspired by the common themes and topics in two different set of video titles for YouTube videos. " 
            f"The first set of video titles were already made by {channel_name} and overperformed for them in the past. These videos are: [{format_titles(titles_of_my_overperformers)}] "
            f"Summarize the common topics and themes that may have made these videos successful but never replicate these ideas exactly. " 
            f"The second set of video titles are not made by {channel_name} but overperformed with their audience. These videos are: [{format_titles(titles_of_related_overperformers)}] "
            f"Do not directly copy the original videos. Make sure that these ideas still align with {channel_name}'s content style and channel. "
        )
    elif inspired_by == 'myOverPerformers':
        inspired_section = (
            f"For your reference, here are titles for some of the best performing videos {channel_name} has made in the past: [{format_titles(titles_of_my_overperformers)}] "
            f"Take these titles as initial inspiration and put them on steroids.  " 
            f"Pitch {channel_name} ideas that are bigger, bolder, faster-paced, more grandiose, more specific, more spectacular, and more unprecedented than any video they have ever made before. "
        )
    elif inspired_by == 'relatedOverPerformers':
        inspired_section = (
            f"The new ideas for {channel_name}'s channel must be inspired by the following set of video titles made by other creators:  [{format_titles(titles_of_my_overperformers)}] "
            f"Take these titles as initial inspiration and put them on steroids. Make sure each idea you pitch is unique and focuses on a different topic or theme. Make sure that these ideas still align with {channel_name}'s content style and channel."
        )
    else:
        inspired_section = (
            f"The new ideas for {channel_name}'s channel must be inspired by the following set of video titles made by other creators: [{format_titles(titles_of_related_overperformers)}] "
            f"Take these titles as initial inspiration and put them on steroids. Make sure each idea you pitch is unique and focuses on a different topic or theme. " 
            f"Make sure that these ideas still align with {channel_name}'s content style and channel. "
        )

    # Constructing the full template
    prompt = f"""
    You will be pitching new video ideas for the YouTube channel of the creator {channel_name}. Your job is to describe what happens in the video in one sentence, NO LONGER THAN 55 WORDS, including the premise, the twist, and the conclusion.
    Here are some details about the YouTube creator {channel_name} and their YouTube channel: {channel_bio}
    {inspired_section}
    Your Response should be a json object, with a single parameter named \""elements\"". 
    \""elements\"" should be a json array with multiple objects that have the fields: 'result' containing a generated concept, 
    'type' is always \""concept\"", and 'id' containing the index of the element. 
    Ensure that the array contains 6 objects.
    """
    prompt += '''\nFor example, your response should look something like: { \""elements\"": [{ \""id\"": 1, \""type\"": \""concept\"", \""result\"": \""story element 1\"" }, { \""id\"": 2, \""type\"": \""concept\"", \""result\"": \""story element 2\""  }... { \""id\"": 6, \""type\"": \""concept\"", \""result\"": \""story element 6\""  }] }'''

    return prompt

def concepts_user_prompt(channel_name, sample_concept, topic=None):
    prompt = f'''Generate 6 unique, original video premises that {channel_name} could make for their YouTube channel.
  A video premise describes what happens in the video in one sentence, including the premise, the twist, and the conclusion, highlighting {channel_name} as the protagonist, a goal, a core conflict, and the key resolution.
  Here is an example of a YouTube video idea that {channel_name} got pitched in the past which they actually made - to give you a sense of the format of idea that {channel_name} likes to be pitched in: {sample_concept}'''
    prompt += f"{channel_name} has requested that the video be based on the following concept '{topic}' so make sure all six video premises are about this concept, with each one tackling the topic from slightly different angles."
    return prompt



def concepts_my_overperformers_system_prompt(channel_name,channel_bio, my_overperformers_titles):
    my_overperformers_titles = "\n".join([f'"{title}"' for title in my_overperformers_titles])
    prompt = f'''
You will be pitching new video ideas for the YouTube channel of the creator {channel_name}. Your job is to describe what happens in the video in one sentence, NO LONGER THAN 55 WORDS, including the premise, the twist, and the conclusion.
Here are some details about the YouTube creator {channel_name} and their YouTube channel: {channel_bio}.
For your reference, here are titles for some of the best performing videos {channel_name} has made in the past:
\""\""\""
{my_overperformers_titles}
\""\""\""    
Take these titles as initial inspiration and put them on steroids. Pitch ideas that are bigger, bolder, faster-paced, more grandiose, more specific, more spectacular, and more unprecedented than any video they have ever made before.
Your Response should be a json object, with a single parameter named \""elements\"". 
\""elements\"" should be a json array with multiple objects that have the fields: 'result' containing a generated concept, 
'type' is always \""concept\"", and 'id' containing the index of the element. 
Ensure that the array contains {count} objects.
'''
    prompt += '''\nFor example, your response should look something like: { \""elements\"": [{ \""id\"": 1, \""type\"": \""concept\"", \""result\"": \""story element 1\"" }, { \""id\"": 2, \""type\"": \""concept\"", \""result\"": \""story element 2\""  }... { \""id\"": 6, \""type\"": \""concept\"", \""result\"": \""story element 6\""  }] }
""}, {""role"": ""user"", ""content"": ""'''

    return prompt

# Avoid ideas that are infeasible or impossible to implement for this YouTube creator.

def concepts_my_overperformers_user_prompt(channel_name, sample_concept):
    return f'''Generate {count} unique, original video premises that {channel_name} could make for their YouTube channel.
  A video premise describes what happens in the video in one sentence, including the premise, the twist, and the conclusion, highlighting {channel_name} as the protagonist, a goal, a core conflict, and the key resolution.
  Here is an example of a YouTube video idea that {channel_name} got pitched in the past which they actually made - to give you a sense of the format of idea that {channel_name} likes to be pitched in: {sample_concept}'''


# Concept threshold is based on 25% quantile of results generated in LABS
thr_concept = {'5-Minute Crafts': {'QCE': 0.7142857142857143, 'AEP': 0.55},
 'Dude Perfect': {'QCE': 1.0, 'AEP': 0.75},
 'Filmcore': {'QCE': 0.0, 'AEP': 0.0},
 'Michelle Khare': {'QCE': 0.5714285714285714, 'AEP': 0.8},
 'MrBeast': {'QCE': 1.0, 'AEP': 0.75},
 'Peter Hollens': {'QCE': 0.8571428571428571, 'AEP': 0.65},
 'PrestonGamez': {'QCE': 0.5714285714285714, 'AEP': 0.8},
 'Rebecca Zamolo': {'QCE': 0.7142857142857143, 'AEP': 0.8},
 'Steven He': {'QCE': 1.0, 'AEP': 0.7},
 'SystemZee': {'QCE': 0.2857142857142857, 'AEP': 0.65}}


overperformers_mean = {'5-Minute Crafts': {'DTE': 0.5835086980920313,
  'QTE': 0.8452380952380952,
  'AEP': 0.4293197278911564},
 'Dude Perfect': {'DTE': 0.9743589743589743,
  'QTE': 0.4285714285714285,
  'AEP': 0.8236111111111111},
 'Filmcore': {'DTE': 0.7362433862433864,
  'QTE': 0.8571428571428571,
  'AEP': 0.5561904761904761},
 'Michelle Khare': {'DTE': 0.9538690476190478,
  'QTE': 1.0,
  'AEP': 0.9285714285714286},
 'MrBeast': {'DTE': 0.8459341397849462,
  'QTE': 0.8392857142857142,
  'AEP': 0.9136363636363637},
 'Peter Hollens': {'DTE': 0.8458791208791209,
  'QTE': 0.2857142857142857,
  'AEP': 0.8625541125541126},
 'PrestonGamez': {'DTE': 0.8575837742504411,
  'QTE': 0.24999999999999997,
  'AEP': 0.5958333333333333},
 'Rebecca Zamolo': {'DTE': 0.7577838827838829,
  'QTE': 0.45238095238095233,
  'AEP': 0.7207482993197278},
 'Steven He': {'DTE': 0.8693581780538303,
  'QTE': 0.0909090909090909,
  'AEP': 0.9333333333333332},
 'SystemZee': {'DTE': 0.8048948948948949,
  'QTE': 0.7142857142857143,
  'AEP': 1.0}}

overperformers_min = {'5-Minute Crafts': {'DTE': 0.5281986531986532,
  'QTE': 0.5714285714285714,
  'AEP': 0.3333333333333333},
 'Dude Perfect': {'DTE': 0.8012820512820512,
  'QTE': 0.2857142857142857,
  'AEP': 0.6666666666666666},
 'Filmcore': {'DTE': 0.6351851851851852,
  'QTE': 0.8571428571428571,
  'AEP': 0.51},
 'Michelle Khare': {'DTE': 0.8333333333333334, 'QTE': 1.0, 'AEP': 0.75},
 'MrBeast': {'DTE': 0.7916666666666666,
  'QTE': 0.7142857142857143,
  'AEP': 0.75},
 'Peter Hollens': {'DTE': 0.7841880341880342,
  'QTE': 0.14285714285714285,
  'AEP': 0.6666666666666666},
 'PrestonGamez': {'DTE': 0.7666666666666666,
  'QTE': 0.14285714285714285,
  'AEP': 0.5},
 'Rebecca Zamolo': {'DTE': 0.673076923076923,
  'QTE': 0.2857142857142857,
  'AEP': 0.6},
 'Steven He': {'DTE': 0.8188405797101449, 'QTE': 0.0, 'AEP': 0.6},
 'SystemZee': {'DTE': 0.6216966966966967,
  'QTE': 0.42857142857142855,
  'AEP': 1.0}}

content_gap_mean = {'5-Minute Crafts': 6.954622079185076,
 'Dude Perfect': 9.907765535126448,
 'Michelle Khare': 14.758179175576979,
 'MrBeast': 27.430985440891487,
 'Peter Hollens': 2.078631236894561,
 'PrestonGamez': 12.503550511804049,
 'Rebecca Zamolo': 27.040477567909633,
 'Steven He': 30.711016027685577,
 'SystemZee': 6.880350045787182}

content_gap_min = {'5-Minute Crafts': 0.8329600043829883,
 'Dude Perfect': 5.9394647052655465,
 'Michelle Khare': 8.565665953384281,
 'MrBeast': 15.726692623147542,
 'Peter Hollens': 0.2661964074837031,
 'PrestonGamez': 3.526638698487902,
 'Rebecca Zamolo': 12.527606191375341,
 'Steven He': 30.19924816157066,
 'SystemZee': 3.501426809312788}

# thr_title = overperformers_mean
thr_title = overperformers_min

concept_AEP_max = {'specifity_score': 4, 'richness_score': 4, 'character_alignment_score': 5, 'sentiment_score': 3, 'social_score': 2, 'visual_score': 1, 'feasibility_score': 1, 'total_score': 20}


# criterion_titles = pd.read_csv('data/criterion_titles.csv').set_index('criterion_id')['title'].to_dict()
# criterion_concepts = pd.read_csv('data/criterion_concepts.csv').set_index('criterion_id')['title'].to_dict()

def gpt_stream_title(system_prompt, user_prompt,loaded_data,filtering=False):
    stream = openai_client.chat.completions.create(
        model = openai_model,
        # model="gpt-4",
        # model = "gpt-4-0125-preview",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        stream=True,
        logit_bias = {
            "25": -100,
            "26": -100,
            "551": -100,
            "2652": -100
        },
        max_tokens = 2000,
        temperature = 0.95,
        presence_penalty = 0,
        frequency_penalty = 0
        )
    counter = 0
    thrs = thr_title[loaded_data['name']]
    creator = loaded_data['name']
    buffer = ""
    title_list = []
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content is not None:
            buffer += content
            
            # Check if there is a newline character in the buffer
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if 'result' in line:
                    print(line)
                    line = re.search(r'"result":\s*"([^"]*)"', line).group(1)   
                else:
                    continue

                if filtering:
                    subcols = st.columns([1,2])
                    with subcols[0]:
                        st.markdown(line)
                    cat_scores_df = util_cats.compute_category_similarity(creator, line)
                    CS_scores_html = "<br>".join(f"{cat_scores_df[f'category_{i}'].iloc[0]}; <strong>content gap score: {np.round(cat_scores_df[f'cat{i}_gap_score'].iloc[0],2)}</strong>" for i in range(1,4))

                     
                    # if passed:
                    counter += 1
                    
                    with subcols[1]:
                        st.markdown(f"""
                        <table style='font-size:small;'>
                        <tr><td><strong>Content Gap Potential:</strong></td><td>{CS_scores_html}</td><td><strong>Total Score:</strong></td><td>{np.round(cat_scores_df['total_score'].iloc[0],2)}</span></td><tr>
                        </table>
                        """, unsafe_allow_html=True)
                        st.markdown("")
                else:
                    st.markdown(line)
                    title_list.append(line)
                    counter += 1
        if counter >= 6:
            return title_list
    return title_list



def gpt_stream_concept(system_prompt, user_prompt,loaded_data,filtering=False):
    channel_name = loaded_data['name']
    stream = openai_client.chat.completions.create(
        model = openai_model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        stream=True,
        max_tokens = 2000,
        temperature = 0.95,
        presence_penalty = 0,
        frequency_penalty = 0
    )
    counter = 0
    buffer = ""
    concept_list = []
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content is not None:
            buffer += content
            # print(buffer)
            while '},' in buffer:
                # Split the buffer at the first newline
                line, buffer = buffer.split('},', 1)
                print(line)
                if 'result' in line:
                    line = re.search(r'"result":\s*"([^"]*)"', line).group(1)
                else:
                    continue
                if filtering:
                    subcols = st.columns(2)
                    with subcols[0]:
                        st.markdown(line)
                    cat_scores_df = util_cats.compute_category_similarity(channel_name, line)
                    CS_scores_html = "<br>".join(f"{cat_scores_df[f'category_{i}'].iloc[0]}; <strong>content gap score: {np.round(cat_scores_df[f'cat{i}_gap_score'].iloc[0],2)}</strong>" for i in range(1,4))

                    # if passed:
                    counter += 1
                    with subcols[1]:
                        st.markdown(f"""
                        <table style='font-size:small;'>
                        <tr><td><strong>Content Gap Potential:</strong></td><td>{CS_scores_html}</td><td><strong>Total Score:</strong></td><td>{np.round(cat_scores_df['total_score'].iloc[0],2)}</span></td></tr>
                        </table>
                        """, unsafe_allow_html=True)
                        st.markdown("")

                else:
                    st.markdown(line)
                    concept_list.append(line)
                    counter += 1
        if counter >= 6:
            return concept_list
    return concept_list

def evaluate_concept(loaded_data,lines,filtering=False):
    channel_name = loaded_data['name']
    concept_list = []
    counter = 0
    for line in lines.split('\n'):
        if filtering:
            subcols = st.columns(2)
            with subcols[0]:
                st.markdown(line)
            cat_scores_df = util_cats.compute_category_similarity(channel_name, line)
            CS_scores_html = "<br>".join(f"{cat_scores_df[f'category_{i}'].iloc[0]}; <strong>content gap score: {np.round(cat_scores_df[f'cat{i}_gap_score'].iloc[0],2)}</strong>" for i in range(1,4))

            
            # if passed:
            counter += 1
            with subcols[1]:
                st.markdown(f"""
                <tr><td><strong>Content Gap Potential:</strong></td><td>{CS_scores_html}</td><td><strong>Total Score:</strong></td><td>{np.round(cat_scores_df['total_score'].iloc[0],2)}</span></td><td>thr={np.round(content_gap_min[channel_name],2)}</td></tr>
                </table>
                """, unsafe_allow_html=True)
                st.markdown("")

        else:
            st.markdown(line)
            concept_list.append(line)
            counter += 1
        if counter >= 6:
            return concept_list
    return concept_list



def llm_run(query,model,client,temperature=0.01, attempt_count=0):
    try:
        chat_completion = client.chat.completions.create(messages=[{"role": "user","content": query}], model=model, temperature=temperature)
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"AEP: Retrying request ({attempt_count}) to LLM, as it failed: {e}")
        if attempt_count < 5:
            return llm_run(query, model, client, temperature, attempt_count + 1)
        else:
            return None
        

def feasibility_evaluator(bio,idea,client=openai_client,model=openai_model):
    query = f"Take a deep breath and answer this query step by step. \
{bio} \
Here is an idea for creating a YouTube video for this YouTube Channel: {idea} \
Answer these questions with Yes or No: \
- Are all elements of the idea possible to implement? (examples of impossible concepts are 'playing a game on another planet' or 'surfing on volcano lava'. However difficult to implement ideas such as 'Helicopter riding' are still possible) \
- Are all elements in the idea controllable by the creator? (examples of non-controllable concepts are 'a dolphin comes out of ocean and wave' or 'escape a real earthquake' because having an earthquake is not controllable, but difficult to control ideas such as 'meeting a celebrity' are still controllable) \
- Is the idea coherent and sensible? (example of non-coherent idea is 'he dog ran, and the cake, then blue sky')\
- Is the idea non-contradictive (example of contradictive idea is 'unseen famous people' or 'unseen stereotypes') \
- Is the idea gramattically correct? \
Your response must be a python list of 'Yes' or 'No' with NO EXTRA TEXT BEFORE OR AFTER THE LIST. An example response is ['Yes','No','Yes','Yes','Yes']"
    return llm_run(query,model=model,client=client,temperature=0.1)

