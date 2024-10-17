import streamlit as st
from streamlit import column_config
import pandas as pd
import time

st.set_page_config(layout="wide")


def sentence_eval(df):
    with st.status("Downloading data..."):
        st.write("Searching for data...")
        time.sleep(2)
        st.write("Found URL.")
        time.sleep(1)
        st.write("Downloading data...")
        time.sleep(1)


st.sidebar.title("Navigation Bar")
st.sidebar.selectbox("Select a model", ["GPT2", "BERT"])
st.sidebar.text_input("Enter OpenAI API key")

st.title("LLM Evaluation tool")
st.write(
    "Choose any input sentences you would like to edit for reevaluation. The choosen sentences will be fed to the probing classifier and the results will be placed."
)


if "df" not in st.session_state:
    st.session_state.df = pd.read_csv("./samples.csv")
    st.session_state.df = st.session_state.df[
        st.session_state.df["set-0"] == "test"
    ].reset_index()
    st.session_state.df = st.session_state.df[["inputs", "label"]].rename(
        columns={"inputs": "Sentence", "label": "Label"}
    )

    st.session_state.df["Sentence"] = st.session_state.df["Sentence"].map(
        lambda x: x.lstrip('"(""``').rstrip('"",)"')
    )
    st.session_state.df[["GPT2", "BERT", "DeBERTa", "Aggregated Results"]] = "20%"

    st.session_state.df = st.session_state.df[
        [
            "Sentence",
            "Label",
            "GPT2",
            "BERT",
            "DeBERTa",
            "Aggregated Results",
        ]
    ]


edited_df = st.data_editor(
    st.session_state.df,
    disabled=("Label", "GPT2", "BERT", "DeBERTa", "Aggregated Results"),
    hide_index=1,
    use_container_width=1,
)


if "clicked" not in st.session_state:
    st.session_state.clicked = False


def click_button():
    st.session_state.clicked = True


if st.session_state.clicked:
    # The message and nested widget will remain on the page
    sentence_eval(st.session_state.df)


if st.button("Proceed with Reevaluation", on_click=click_button):
    pass
