import streamlit as st
import util.util_demo as util

creators = ["Michelle Khare",
    "MrBeast",
    "Steven He",
    "SystemZee",
    "Dude Perfect",
    "5-Minute Crafts",
    "Rebecca Zamolo",
    "Peter Hollens",
    "PrestonGamez",
    "Filmcore",
    ]

thr_title = util.thr_title

st.set_page_config(layout="wide")


def main():
    
    st.title("Idea Ranking & Filtering")
    
            
    tab1, tab2, tab3 = st.tabs(["Online Concept Test", "Concept Test","Online Title Test"])
      
    with tab3:
        cols_header =st.columns(2)
        with cols_header[0]:
            creator = st.selectbox('Select Creator ',creators)
            loaded_data = util.read_json(f"./data/channels/{creator}.json")
            channel_bio = util.get_bio(creator)
            # my_overperformers_titles = util.overperformers.groupby(0)[1].apply(list).to_dict()[creator]
            titles_of_my_overperformers = loaded_data["overperformers"]
            titles_of_related_overperformers = loaded_data["relatedoverperformers"]
            inspired_by = None
            topic = None
            # system_prompt = util.titles_my_overperformers_system_prompt(creator,channel_bio, my_overperformers_titles,loaded_data)
            system_prompt = util.titles_system_prompt(creator,channel_bio,inspired_by,titles_of_my_overperformers,titles_of_related_overperformers)
            # user_prompt = util.titles_my_overperformers_user_prompt(creator)
            user_prompt = util.titles_user_prompt(creator,topic)
        with cols_header[1]:
            gen_path_titles = st.selectbox('Select Generation Path ',['Filtered','Non-filtered'])
        if st.button('Run Title Generation'):
            util.gpt_stream_title(system_prompt,user_prompt,loaded_data,filtering=(gen_path_titles=='Filtered'))

    with tab1:
        cols_header =st.columns(2)
        with cols_header[0]:
            creator_c = st.selectbox('Select Creator',creators)
            loaded_data = util.read_json(f"./data/channels/{creator_c}.json")
            channel_bio = util.get_bio(creator_c)
            titles_of_my_overperformers = loaded_data["overperformers"]
            titles_of_related_overperformers = loaded_data["relatedoverperformers"]
            inspired_by = None
            topic = None
            # my_overperformers_titles = util.overperformers.groupby(0)[1].apply(list).to_dict()[creator_c]
            system_prompt = util.concepts_system_prompt(creator_c,channel_bio,inspired_by,titles_of_my_overperformers,titles_of_related_overperformers)
            user_prompt = util.concepts_user_prompt(creator_c, loaded_data['concepts'],topic)
        with cols_header[1]:
            gen_path_concepts = st.selectbox('Select Generation Path',['Filtered','Non-filtered'])
        if st.button('Run Concept Generation'):
            util.gpt_stream_concept(system_prompt, user_prompt,loaded_data,filtering=(gen_path_concepts=='Filtered'))
            # util.gpt_stream_concept(loaded_data,filtering=(gen_path_concepts=='Filtered'))

    with tab2:
        cols_header =st.columns(2)
        with cols_header[0]:
            creator_c = st.selectbox('Select a Creator',creators)
            loaded_data = util.read_json(f"./data/channels/{creator_c}.json")
            channel_bio = util.get_bio(creator_c)
            titles_of_my_overperformers = loaded_data["overperformers"]
            titles_of_related_overperformers = loaded_data["relatedoverperformers"]
            inspired_by = None
            topic = None
            # my_overperformers_titles = util.overperformers.groupby(0)[1].apply(list).to_dict()[creator_c]
            system_prompt = util.concepts_system_prompt(creator_c,channel_bio,inspired_by,titles_of_my_overperformers,titles_of_related_overperformers)
            user_prompt = util.concepts_user_prompt(creator_c, loaded_data['concepts'],topic)
        # with cols_header[1]:
        #     gen_path_concepts = st.selectbox('Select Generation Path',['Filtered','Non-filtered'])
        lines = st.text_area('Enter concepts separated by single newline character.')
        if st.button('Run Concept Evaluation'):
            util.evaluate_concept(loaded_data,lines,filtering=True)
            # util.gpt_stream_concept(system_prompt, user_prompt,loaded_data,filtering=(gen_path_concepts=='Filtered'))

if __name__ == "__main__":
    main()