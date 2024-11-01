import streamlit as st
from streamlit import column_config
import pandas as pd
import time
import glob
import sys
import os

sys.path.append("../backend/")
sys.path.append("../../src/")
import backend

st.set_page_config(layout="wide")


def sentence_eval(df):
    temp_data = []

    for index, item in enumerate(df["modified_sentence"]):
        temp_data.append(
            {
                "inputs": f"('{item}',)",
                "context": "",
                "topic": "",
                "org_label": 0,
                "set-0": "test",
                "id": index,
                "label": 0,
            }
        )

    temp_df = pd.DataFrame(
        temp_data,
        columns=["inputs", "context", "topic", "org_label", "set-0", "id", "label"],
    )

    temp_df.to_csv(
        f"../../data/holmes/{selected_task}/modified_samples.csv",
        index=False,
    )

    return backend.Backend()


# Define the directory and task folders
directory_path = "../../data/holmes/"
folders = [
    f
    for f in os.listdir(directory_path)
    if os.path.isdir(os.path.join(directory_path, f))
]

# Sidebar for task selection
st.sidebar.title("Navigation Bar")
st.sidebar.write("Please choose a probing task here")
selected_task = st.sidebar.selectbox("Choose a probing task", sorted(folders))
selected_model = st.sidebar.multiselect(
    "Not implemented yet...",
    default=[
        "EleutherAI/pythia-12b-deduped",
        "facebook/bart-base",
        "microsoft/Orca-2-13b",
        "EleutherAI/pythia-6.9b-deduped",
    ],
    options=[
        "google/flan-ul2",
        "google/flan-t5-xxl",
        "google/t5-xxl-lm-adapt",
        "lmsys/vicuna-13b-v1.5",
        "meta-llama/Llama-2-70b-chat-hf",
        "ibm/labradorite-13b",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "EleutherAI/pythia-12b-deduped",
        "facebook/bart-base",
        "microsoft/Orca-2-13b",
        "EleutherAI/pythia-6.9b-deduped",
        "google/ul2",
        "google/flan-t5-xl",
        "google/t5-xl-lm-adapt",
        "google/electra-base-discriminator",
        "databricks/dolly-v2-12b",
        "EleutherAI/pythia-12b",
        "allenai/tulu-2-13b",
        "EleutherAI/pythia-6.9b",
        "microsoft/deberta-v3-base",
        "EleutherAI/pythia-2.8b-deduped",
        "meta-llama/Llama-2-70b-hf",
        "allenai/tulu-2-dpo-13b",
        "WizardLM/WizardLM-13B-V1.2",
        "microsoft/deberta-base",
        "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-2.8b",
        "allenai/tulu-2-70b",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "albert-base-v2",
        "allenai/tk-instruct-11b-def",
        "allenai/tulu-2-dpo-70b",
        "google/flan-t5-large",
        "google/t5-base-lm-adapt",
        "google/flan-t5-base",
        "EleutherAI/pythia-1b-deduped",
        "meta-llama/Llama-2-7b-hf",
        "EleutherAI/pythia-1.4b-deduped",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "bert-base-uncased",
        "mistralai/Mistral-7B-v0.1",
        "meta-llama/Llama-2-7b-chat-hf",
        "ibm/merlinite-7b",
        "roberta-base",
        "google/t5-large-lm-adapt",
        "mistralai/Mixtral-8x7B-v0.1",
        "gpt2",
        "EleutherAI/pythia-410m",
        "google/flan-t5-small",
        "google/t5-small-lm-adapt",
        "EleutherAI/pythia-410m-deduped",
        "Glove.840B",
        "EleutherAI/pythia-160m-deduped",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-70m-deduped",
    ],
    max_selections=4,
)
openai_prompt = st.sidebar.text_area(
    label="OpenAI Prompt",
    value="Is this sentence negated or not, if it is please answer with a 1 else with a 0",
)


# Callback function to update data based on the selected probing task
def update_task():
    st.session_state.df = pd.read_csv(f"../../data/holmes/{selected_task}/samples.csv")
    path = glob.glob(
        f"../../results/holmes/{selected_task}/microsoft__deberta-v3-base/full/NONE/**/**/0/done/preds.csv"
    )
    st.session_state.df = st.session_state.df[
        st.session_state.df["set-0"] == "test"
    ].reset_index()
    st.session_state.df = st.session_state.df[["inputs", "label"]].rename(
        columns={"inputs": "Sentence", "label": "Label"}
    )

    files = pd.concat([pd.read_csv(file) for file in path])
    st.session_state.model0 = files.groupby("Unnamed: 0")["pred"].mean().reset_index()

    st.session_state.df["Sentence"] = st.session_state.df["Sentence"].map(
        lambda x: x.lstrip('"(""``').rstrip('"",)"')
    )
    st.session_state.df["Bart Base"] = st.session_state.model0["pred"].apply(
        lambda x: "{:.2f}%".format(x * 100)
    )
    st.session_state.df = st.session_state.df[["Sentence", "Label", "Bart Base"]]


def investigate_models():
    for i in selected_model:
        probing_command = f"python3 investigate.py --model_name {i} --version holmes  --cuda_visible_devices 0,1 --dump_preds --in_filter {selected_task}"
        os.chdir("../../src/")
        os.system(probing_command)
        os.chdir("../extension/frontend/")


# Button to load data for the selected task
if st.sidebar.button("Load Task Data"):
    update_task()
    # investigate_models()

# Load data initially if not already loaded
st.title("LLM Evaluation tool")

if "df" not in st.session_state:
    st.write(
        "Choose any input sentences you would like to edit for reevaluation. The chosen sentences will be fed to the probing classifier and the results will be placed."
    )

    # Call the update_task function to load the initial dataframe
    update_task()

# Data editor for displaying and editing the DataFrame
edited_df = st.data_editor(
    st.session_state.df,
    disabled=("Label", "Bart Base"),
    hide_index=1,
    use_container_width=1,
    column_config={
        "Sentence": st.column_config.Column(
            "Sentence",
            help="You are able to change each input sentence for reevaluation",
        ),
        "Label": st.column_config.Column(
            "Label",
            help="Here you can see the preannotated label aka the ground truth",
        ),
    },
)


def get_changed_rows_df(original_df, edited_df):
    changes = original_df[original_df["Sentence"] != edited_df["Sentence"]]
    changes["modified_sentence"] = edited_df.loc[changes.index, "Sentence"]
    return changes[["Sentence", "modified_sentence"]]


if "clicked" not in st.session_state:
    st.session_state.clicked = False


def click_button():
    st.session_state.clicked = True


if st.session_state.clicked:
    changes_df = get_changed_rows_df(st.session_state.df, edited_df)
    if not changes_df.empty:
        st.write("Changed rows:")
        evaluated_df = sentence_eval(changes_df).drop(
            columns="label"
        )  # Get evaluated DataFrame

        # Rename columns in evaluated_df to match st.session_state.df for consistency
        evaluated_df = evaluated_df.rename(
            columns={"instance": "Sentence", "pred": "Bart Base"}
        )

        evaluated_df["Sentence"] = evaluated_df["Sentence"].map(
            lambda x: str(x).lstrip('"(""``').rstrip('"",)"')
        )
        evaluated_df["Bart Base"] = evaluated_df["Bart Base"].apply(
            lambda x: "{:.2f}%".format(x * 100)
        )

        # Update edited_df in st.data_editor with the new rows
        edited_df = st.dataframe(
            evaluated_df,
            hide_index=1,
            use_container_width=1,
        )
    else:
        st.write("No changes detected.")

if st.button("Proceed with Reevaluation", on_click=click_button):
    pass
