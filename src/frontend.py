import streamlit as st
import pandas as pd
from streamlit import column_config
import glob
import os
import openai

from backend import Backend

st.set_page_config(layout="wide")


def sentence_eval(df, selected_models=["microsoft/deberta-v3-base"]):
    temp_data = []

    for index, (sentence, modified_label) in enumerate(
        zip(df["modified_sentence"], df["modified_label"])
    ):
        escaped_item = sentence.replace("'", "\\'")
        temp_data.append(
            {
                "inputs": f"('{escaped_item}',)",
                "context": "",
                "topic": "",
                "org_label": 0,
                "set-0": "test",
                "id": index,
                "label": modified_label,
            }
        )

    temp_df = pd.DataFrame(
        temp_data,
        columns=["inputs", "context", "topic", "org_label", "set-0", "id", "label"],
    )

    temp_df.to_csv(
        f"./data/holmes/{selected_task}/modified_samples.csv",
        index=False,
    )

    return Backend(probing_task=f"{selected_task}", selected_models=selected_models)


# Define the directory and task folders
directory_path = "./data/holmes/"
folders = [
    f
    for f in os.listdir(directory_path)
    if os.path.isdir(os.path.join(directory_path, f))
]

# Sidebar for task selection
st.sidebar.title("Navigation Bar")
st.sidebar.write("Please choose a probing task here")
selected_task = st.sidebar.selectbox("Choose a probing task", sorted(folders))
st.session_state.selected_model = st.sidebar.multiselect(
    "Please select at most 4 models from here",
    default=["microsoft/deberta-v3-base", "albert/albert-base-v2"],
    options=["microsoft/deberta-v3-base", "albert/albert-base-v2"],
    max_selections=4,
)
use_openai_response = st.sidebar.checkbox("Get OpenAI Response for Modified Sentences")
if use_openai_response:
    api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    user_prompt_template = st.sidebar.text_area(
        label="OpenAI Prompt",
        value="Is the following sentence grammatically acceptable or no?: {sentence}",
        help="The {sentence} placeholder is required for every prompt you might create. you can add it anywhere in your prompt.",
    )
    if "{sentence}" not in user_prompt_template:
        st.error(
            "Error: The prompt template must contain '{sentence}'. Please include it in your template."
        )


def get_openai_responses(api_key, user_prompt_template, changes_df):
    client = openai.OpenAI(api_key=api_key)
    responses = []

    for index, row in changes_df.iterrows():
        modified_sentence = row["modified_sentence"]

        prompt = user_prompt_template.format(sentence=modified_sentence)

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
            )

            result = response.choices[0].message.content.strip()

            responses.append(
                {
                    "Modified Sentence": modified_sentence,
                    "OpenAI Response": result,
                }
            )

        except Exception as e:
            st.error(f"Error fetching response for sentence '{modified_sentence}': {e}")

    # Convert the list of responses to a DataFrame
    return pd.DataFrame(responses)


def load_model_dfs(models):
    list_of_preds_dfs = []
    for i in models:
        model_name = i.replace("/", "__")
        path = glob.glob(
            f"./results/holmes/{selected_task}/{model_name}/full/NONE/**/**/0/done/preds.csv"
        )
        files = pd.concat([pd.read_csv(file) for file in path])

        def modify_prediction(row):
            if row["label"] == 0 and row["pred"] == 0:
                return 1.0
            elif row["label"] == 1 and row["pred"] == 1:
                return 1.0
            elif row["label"] == 0 and row["pred"] == 1:
                return 0.0
            elif row["label"] == 1 and row["pred"] == 0:
                return 0.0
            else:
                return row["pred"]

        files["modified_pred"] = files.apply(modify_prediction, axis=1)

        st.session_state.model = (
            files.groupby("Unnamed: 0")["modified_pred"].mean().reset_index()
        )
        list_of_preds_dfs.append(st.session_state.model["modified_pred"])

    return list_of_preds_dfs


# Callback function to update data based on the selected probing task
def update_task():
    st.session_state.df = pd.read_csv(f"./data/holmes/{selected_task}/samples.csv")
    st.session_state.df = st.session_state.df[
        st.session_state.df["set-0"] == "test"
    ].reset_index()
    st.session_state.df = st.session_state.df[["inputs", "label"]].rename(
        columns={"inputs": "Sentence", "label": "Label"}
    )

    st.session_state.df["Sentence"] = st.session_state.df["Sentence"].map(
        lambda x: x.lstrip('"(""``').rstrip('"",)"')
    )
    for i in range(len(st.session_state.selected_model)):
        st.session_state.df[f"{st.session_state.selected_model[i]}"] = load_model_dfs(
            st.session_state.selected_model
        )[i]

    # Define averaging logic
    def average_results(row):
        # Collect numerical results from each model for the current row
        model_results = [row[model] for model in st.session_state.selected_model]
        # Calculate the mean of the model results
        return sum(model_results) / len(model_results)

    # Apply averaging function to each row
    st.session_state.df["Aggregated Results"] = st.session_state.df.apply(
        average_results, axis=1
    )


def investigate_models():
    for i in st.session_state.selected_model:
        probing_command = f"python3 investigate.py --model_name {i} --version holmes  --cuda_visible_devices 0,1 --dump_preds --in_filter {selected_task}"
        os.chdir("../../src/")
        os.system(probing_command)
        os.chdir("../extension/frontend/")


def calculate_end_percentages():
    overall_percentages = {
        model: (st.session_state.df[model].astype(float) / 100).mean() * 100
        for model in st.session_state.selected_model + ["Aggregated Results"]
    }

    overall_percentages_formatted = {
        model: "{:.2f}%".format(overall_percentages[model])
        for model in overall_percentages
    }

    st.session_state.overall_percentages_df = pd.DataFrame(
        overall_percentages_formatted, index=["Overall Percentage"]
    )

    for model in st.session_state.selected_model + ["Aggregated Results"]:
        st.session_state.df[model] = st.session_state.df[model].apply(
            lambda x: "{:.2f}%".format(x * 100)
        )


if st.sidebar.button("Load Task Data"):
    update_task()
    calculate_end_percentages()
    # investigate_models()

# Load data initially if not already loaded
st.title("LLM Evaluation tool")

if "df" not in st.session_state:
    st.write(
        "Choose any input sentences you would like to edit for reevaluation. The chosen sentences will be fed to the probing classifier and the results will be placed."
    )

    # Call the update_task function to load the initial dataframe
    update_task()
    calculate_end_percentages()

edited_df = st.data_editor(
    st.session_state.df,
    disabled=st.session_state.selected_model + ["Aggregated Results"],
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

percentages = st.dataframe(st.session_state.overall_percentages_df)


def get_changed_rows_df(original_df, edited_df):
    changes = original_df[original_df["Sentence"] != edited_df["Sentence"]]
    changes["modified_sentence"] = edited_df.loc[changes.index, "Sentence"]
    changes["modified_label"] = edited_df.loc[changes.index, "Label"]
    return changes[["Sentence", "modified_sentence", "Label", "modified_label"]]


if "clicked" not in st.session_state:
    st.session_state.clicked = False


def click_button():
    st.session_state.clicked = True


if st.button("Proceed with Reevaluation", on_click=click_button):
    pass

if st.session_state.clicked:
    changes_df = get_changed_rows_df(st.session_state.df, edited_df)
    if not changes_df.empty:
        st.write("Changed rows:")

        if use_openai_response:
            if "{sentence}" not in user_prompt_template:
                st.error(
                    "Error: The prompt template must contain '{sentence}'. Please include it in your template."
                )
            else:
                openai_responses_df = get_openai_responses(
                    api_key, user_prompt_template, changes_df
                )

                st.write("OpenAI Responses for Modified Sentences:")
                st.dataframe(
                    openai_responses_df, hide_index=True, use_container_width=True
                )
        else:
            st.write("OpenAI response generation is disabled.")

        # Run evaluation for multiple models
        model_predictions = sentence_eval(
            changes_df, selected_models=st.session_state.selected_model
        )

        # Initialize a list to store model data for merging
        dfs_to_merge = []

        # Loop over each model to process and prepare its results
        for model_name, model_df in model_predictions.items():
            # Rename columns and clean the DataFrame
            model_df = model_df.rename(
                columns={
                    "instance": "Sentence",
                    "label": "Label",
                    "pred": f"{model_name} Prediction",
                }
            ).drop(["loss"], axis=1)

            # Format the columns
            model_df["Sentence"] = model_df["Sentence"].map(
                lambda x: str(x).lstrip('"(""``').rstrip('"",)"')
            )
            model_df[f"{model_name} Prediction"] = model_df[
                f"{model_name} Prediction"
            ].apply(lambda x: "{:.2f}%".format(x * 100))

            # Append to list for merging
            dfs_to_merge.append(model_df)

        # Merge all DataFrames on the 'Sentence' column
        if dfs_to_merge:
            combined_df = dfs_to_merge[0]
            for df in dfs_to_merge[1:]:
                combined_df = combined_df.merge(
                    df, on=["Sentence", "Label"], how="outer"
                )

            columns_order = ["Sentence", "Label"] + [
                col for col in combined_df.columns if col not in ["Sentence", "Label"]
            ]
            combined_df = combined_df[columns_order]

            # Display the combined DataFrame
            st.write("Combined Model Predictions:")
            st.dataframe(
                combined_df,
                hide_index=True,
                use_container_width=True,
            )

        st.session_state.evaluated_df = combined_df

        chart_data = combined_df.drop(["Label"], axis=1).set_index("Sentence")
        chart_data = chart_data.applymap(
            lambda x: float(x.strip("%")) / 100
            if isinstance(x, str) and "%" in x
            else x
        )

        # Plot all model predictions in a single chart
        st.write("Model Predictions Visualization:")
        st.bar_chart(chart_data, stack=False)

    else:
        st.write("No changes detected.")
