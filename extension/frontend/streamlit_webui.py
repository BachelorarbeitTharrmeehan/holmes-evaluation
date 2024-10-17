import streamlit as st
import pandas as pd

df = pd.read_csv("../../results/preds.csv")


def on_data_change():
    modified_data = st.session_state["edited_df"]["edited_rows"]
    st.write(modified_data)
    sentences = []
    for i in modified_data:
        sentences.append(modified_data[i]["Sentence"])
    sentence_to_reevaluate = sentences.pop()
    st.write(sentence_to_reevaluate)


def main():
    st.title("LLM Evaluation Web Application")
    st.write("This is a Streamlit web application.")
    st.data_editor(
        {"Sentence": df["0"], "Prediction": df["1"], "BERT": df["3"]},
        disabled=("Prediction", "BERT"),
        hide_index=1,
        use_container_width=1,
        on_change=on_data_change,
        key="edited_df",
    )


if __name__ == "__main__":
    main()
