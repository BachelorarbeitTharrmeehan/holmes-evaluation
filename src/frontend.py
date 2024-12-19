import streamlit as st
import pandas as pd
import numpy as np
from streamlit import column_config
import glob
import os
import subprocess
import openai
import yaml

from backend import Backend
from config_extension import EXTENSION_CONFIG


st.set_page_config(layout="wide")


def sentence_eval(df, selected_models=["microsoft/deberta-v3-base"]):
    temp_data = []

    sentence_columns = [
        col for col in df.columns if col.startswith("modified_Sentence")
    ]

    for index, row in df.iterrows():
        escaped_sentences = [row[col].replace("'", "\\'") for col in sentence_columns]
        escaped_context = (
            row["Context"].replace("'", "\\'") if "Context" in df.columns else ""
        )

        if len(escaped_sentences) == 1:
            inputs = f"('{escaped_sentences[0]}',)"
        else:
            inputs = "(" + ", ".join(f"('{s}')" for s in escaped_sentences) + ")"

        temp_data.append(
            {
                "inputs": inputs,
                "context": escaped_context,
                "topic": "",
                "org_label": 0,
                "set-0": "test",
                "id": index,
                "label": row.get(
                    "modified_Label", 0
                ),  # Use default 0 if 'modified_label' is missing
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

    return Backend(
        probing_task=f"{selected_task}",
        selected_models=selected_models,
        probe_task_type=probe_task_type,
    )


def get_subfield_and_phenomena(task):
    leaderboard = pd.read_csv("./data/leaderboards/holmes.csv")
    filtered_df = leaderboard[leaderboard["probing dataset"].str.contains(task)]
    result_df = filtered_df[["linguistic subfield", "linguistic phenomena"]]
    return result_df


# Define the directory and task folders
directory_path = "./data/holmes/"
folders = [
    f
    for f in os.listdir(directory_path)
    if os.path.isdir(os.path.join(directory_path, f))
]

# Sidebar for task selection
st.sidebar.title("Navigation Bar")
on = st.sidebar.toggle("Load Custom Configuration")
if on:
    if EXTENSION_CONFIG["selected_task"]:
        selected_task = EXTENSION_CONFIG["selected_task"]
else:
    selected_task = st.sidebar.selectbox(
        "Please choose a probing task from here", sorted(folders)
    )
subfield_df = get_subfield_and_phenomena(selected_task)
if not subfield_df.empty:
    st.sidebar.markdown(
        f"**Linguistic Subfield:** {subfield_df.iloc[0]['linguistic subfield']}"
    )
    st.sidebar.markdown(
        f"**Linguistic Phenomena:** {subfield_df.iloc[0]['linguistic phenomena']}"
    )
else:
    st.sidebar.markdown("**Linguistic Subfield:** Not available")
    st.sidebar.markdown("**Linguistic Phenomena:** Not available")

if on:
    if EXTENSION_CONFIG["selected_models"]:
        st.session_state.selected_model = EXTENSION_CONFIG["selected_models"]
        precision = "full"
else:
    st.session_state.selected_model = st.sidebar.multiselect(
        "Please select at most 4 models from here",
        default=[
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        ],
        options=[
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
            "Qwen/Qwen2.5-1.5B",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            "Qwen/Qwen2.5-3B",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct-AWQ",
            "Qwen/Qwen2.5-7B",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "albert/albert-base-v2",
        ],
        max_selections=4,
    )
    option_map = {
        "full": "Full",
        "half": "Half",
        "eight_bit": "8Bit",
        "four_bit": "4Bit",
    }
    precision = st.sidebar.segmented_control(
        "Model Precision",
        options=option_map.keys(),
        default="full",
        format_func=lambda option: option_map[option],
        selection_mode="single",
    )

config_file_path = f"./data/flash-holmes/{selected_task}/config-none.yaml"
fallback_config_file_path = f"./data/flash-holmes/{selected_task}/config-bi-none.yaml"

if os.path.exists(config_file_path):
    config_file_to_use = config_file_path
else:
    config_file_to_use = fallback_config_file_path

try:
    with open(config_file_to_use, "r") as file:
        config_data = yaml.safe_load(file)
except FileNotFoundError:
    st.error(f"Neither {config_file_path} nor {fallback_config_file_path} was found.")
    config_data = {}


probe_task_type = config_data.get("probe_task_type", None)

use_openai_response = st.sidebar.toggle("OpenAI Response for Modified Sentences")
if use_openai_response:
    api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    task_prompts = {
        "SemAntoNeg": "For the given sentences:\n"
        "1. {sentence_1}\n"
        "2. {sentence_2}\n"
        "The task probes language models' understanding of negation and antonymy. Determine whether sentence 2 is the correct semantic counterpart to sentence 1. Provide detailed feedback on whether negation markers and antonym substitutions are applied correctly and if the sentences are semantically equivalent.",
        "task2": "Analyze the linguistic structure of the following sentence: {sentence_1}",
        "task3": "Determine if the following sentence conveys ambiguity: {sentence_1}",
        # Add more prompts per task as required, optional: read in file with all the data to reduce size
    }

    if "Selected" not in st.session_state.df.columns:
        st.session_state.df["Selected"] = False

    if selected_task in task_prompts:
        default_prompt = task_prompts[selected_task]
    else:
        default_prompt = "Please evaluate the following sentence: {sentence_1}"

    user_prompt_template = st.sidebar.text_area(
        label="OpenAI Prompt",
        value=default_prompt,
        help="The {sentence_1}, {sentence_2}, etc. placeholders are required for every prompt you might create. you can add them anywhere in your prompt.",
    )


def get_openai_responses(api_key, user_prompt_template, changes_df):
    client = openai.OpenAI(api_key=api_key)
    responses = []

    sentence_columns = [col for col in changes_df.columns if col.startswith("Sentence")]

    for index, row in changes_df.iterrows():
        # Map placeholders dynamically to sentence values
        sentence_placeholders = {
            f"sentence_{i+1}": row[col].replace("'", "\\'")
            for i, col in enumerate(sentence_columns)
        }

        try:
            prompt = user_prompt_template.format(**sentence_placeholders)
        except KeyError as e:
            st.error(
                f"Error: Missing placeholder for {str(e)}. Please include all required placeholders in your prompt."
            )
            continue

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
                    "Modified Sentences": list(sentence_placeholders.values()),
                    "OpenAI Response": result,
                }
            )

        except Exception as e:
            st.error(
                f"Error fetching response for sentences '{sentence_placeholders}': {e}"
            )

    # Convert the list of responses to a DataFrame
    return pd.DataFrame(responses)


def load_model_dfs(models):
    import os

    list_of_preds_dfs = []
    for i in models:
        model_name = i.replace("/", "__")
        path = glob.glob(
            f"./results/flash-holmes/{selected_task}/{model_name}/{precision}/NONE/**/**/0/done/preds.csv"
        )
        model_folder_path = os.path.join(
            "./results/flash-holmes", selected_task, model_name
        )

        if not os.path.exists(model_folder_path):
            st.warning(
                f"Model folder for {model_name} not found. Investigating model..."
            )
            investigate_models()

        try:
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

        except Exception as e:
            st.error(f"Error loading predictions for {model_name}: {e}")

    return list_of_preds_dfs


def update_task():
    st.session_state.df = pd.read_csv(
        f"./data/flash-holmes/{selected_task}/samples.csv"
    )
    st.session_state.df = st.session_state.df[
        st.session_state.df["set-0"] == "test"
    ].reset_index()
    st.session_state.df = st.session_state.df[["inputs", "context", "label"]].rename(
        columns={"inputs": "Sentence", "label": "Label", "context": "Context"}
    )

    def split_tuple_into_columns(input_value):
        try:
            evaluated = eval(input_value)
            evaluated = tuple(str(item) for item in evaluated)
            return evaluated
        except Exception as e:
            st.error(f"Error evaluating input: {input_value} - {e}")
            return ("Error",)  # Return a placeholder in case of error

    # Apply the function to evaluate the "Sentence" column
    st.session_state.df["Sentence"] = st.session_state.df["Sentence"].map(
        split_tuple_into_columns
    )

    # Expand the tuple into multiple columns
    max_tuple_length = st.session_state.df["Sentence"].map(len).max()
    for i in range(max_tuple_length):
        st.session_state.df[f"Sentence {i+1}"] = st.session_state.df["Sentence"].map(
            lambda x: x[i] if i < len(x) else None
        )

    # Drop the original "Sentence" column if it's no longer needed
    st.session_state.df.drop(columns=["Sentence"], inplace=True)

    # Reorder columns: Sentence parts, Context, Label, then others
    sentence_columns = [f"Sentence {i+1}" for i in range(max_tuple_length)]
    st.session_state.df = st.session_state.df[
        sentence_columns
        + ["Context"]
        + ["Label"]
        + [
            col
            for col in st.session_state.df.columns
            if col not in sentence_columns + ["Context", "Label"]
        ]
    ].dropna(how="all", axis=1)

    for i in range(len(st.session_state.selected_model)):
        st.session_state.df[f"{st.session_state.selected_model[i]}"] = load_model_dfs(
            st.session_state.selected_model
        )[i]

    # Average and Standard Deviation
    def calculate_statistics(row):
        # Collect numerical results from each model for the current row
        model_results = [row[model] for model in st.session_state.selected_model]
        # Calculate the mean and standard deviation of the model results
        mean_result = sum(model_results) / len(model_results)
        std_dev_result = np.std(model_results)
        return mean_result, std_dev_result

    # Apply the function to each row and store results in new columns
    st.session_state.df["Aggregated Results"], st.session_state.df["Std Deviation"] = (
        zip(*st.session_state.df.apply(calculate_statistics, axis=1))
    )


def investigate_models():
    base_path = "./src/"
    for model in st.session_state.selected_model:
        try:
            probing_command = [
                "python3",
                "investigate.py",
                "--model_name",
                model,
                "--version",
                "flash-holmes",
                "--cuda_visible_devices",
                "0,1",
                "--dump_preds",
                "--in_filter",
                selected_task,
            ]

            # Execute the command in the appropriate directory
            result = subprocess.run(
                probing_command,
                cwd=base_path,  # Specify the working directory
                capture_output=True,  # Capture stdout and stderr
                text=True,  # Decode output as text
                check=True,  # Raise an exception if the command fails
            )

            # Log the output
            st.write(f"Model {model} investigated successfully.")
            st.text(result.stdout)

        except subprocess.CalledProcessError as e:
            # Handle command execution errors
            st.error(f"Error investigating model {model}: {e}")
            st.text(e.stderr)
        except Exception as ex:
            # Handle other errors
            st.error(f"Unexpected error: {ex}")


def is_label_binary():
    return st.session_state.df["Label"].isin([0, 1]).all()


def calculate_label_percentages():
    """
    Calculate the average percentage for every label per model in the DataFrame.
    The percentages are stored and displayed as a new DataFrame.
    """
    # Dictionary to store label-wise percentages for each model
    label_percentages = {}

    # Compute the mean for each model grouped by 'Label'
    for model in st.session_state.selected_model + ["Aggregated Results"]:
        label_percentages[model] = st.session_state.df.groupby("Label")[model].mean()

    # Combine the results into a single DataFrame
    label_percentages_df = pd.DataFrame(label_percentages)

    models_only_columns = [
        col for col in label_percentages_df.columns if col not in ["Aggregated Results"]
    ]

    label_percentages_df["Std Deviation"] = label_percentages_df[
        models_only_columns
    ].std(axis=1, ddof=0)

    # Format the percentages as strings with two decimal places
    if is_label_binary():
        formatted_label_percentages_df = label_percentages_df.applymap(
            lambda x: x * 100 if not pd.isna(x) else "N/A"
        )
    else:
        formatted_label_percentages_df = label_percentages_df

    # Save the formatted results to the session state
    st.session_state.label_percentages_df = formatted_label_percentages_df


def calculate_end_percentages():
    overall_percentages = {
        model: (st.session_state.df[model].astype(float) / 100).mean() * 100
        for model in st.session_state.selected_model + ["Aggregated Results"]
    }

    st.session_state.overall_percentages_df = pd.DataFrame(
        overall_percentages, index=["Overall Percentage"]
    )

    models_only_columns = [
        col
        for col in st.session_state.overall_percentages_df.columns
        if col not in ["Aggregated Results"]
    ]

    st.session_state.overall_percentages_df["Std Deviation"] = (
        st.session_state.overall_percentages_df[models_only_columns].std(axis=1, ddof=0)
    )
    if is_label_binary():
        for model in (
            st.session_state.selected_model + ["Aggregated Results"] + ["Std Deviation"]
        ):
            st.session_state.df[model] = st.session_state.df[model].apply(
                lambda x: x * 100
            )
            st.session_state.overall_percentages_df[model] = (
                st.session_state.overall_percentages_df[model].apply(lambda x: x * 100)
            )


if st.sidebar.button("Load Task Data"):
    update_task()
    calculate_label_percentages()
    calculate_end_percentages()
    # investigate_models()

# Load data initially if not already loaded
st.title("LLM Evaluation tool")

if "df" not in st.session_state:
    st.write(
        "Choose any input sentences you would like to edit for reevaluation. The chosen sentences will be fed to the probing classifier and the results will be shown at the bottom."
    )

    # Call the update_task function to load the initial dataframe
    update_task()
    calculate_label_percentages()
    calculate_end_percentages()

edited_df = st.data_editor(
    st.session_state.df,
    column_order=["Selected"]
    + [col for col in st.session_state.df.columns if col.startswith("Sentence")]
    + ["Label"]
    + ["Std Deviation"]
    + ["Aggregated Results"]
    + st.session_state.selected_model,
    disabled=st.session_state.selected_model
    + ["Aggregated Results"]
    + ["Std Deviation"],
    hide_index=1,
    use_container_width=1,
    column_config={
        **{
            model: st.column_config.NumberColumn(
                format="%.2f%%",
                help=f"Predictions for {model}",
            )
            for model in st.session_state.selected_model
        },
        "Sentence": st.column_config.Column(
            "Sentence",
            help="You are able to change each input sentence for reevaluation",
        ),
        "Label": st.column_config.Column(
            "Label",
            help="Here you can see the preannotated label aka the ground truth",
        ),
        "Aggregated Results": st.column_config.NumberColumn(
            "Aggregated Results",
            help="Here you are able to see the percentage of correct predictions over all models",
            format="%.2f%%",
        ),
        "Std Deviation": st.column_config.NumberColumn(
            "Std Deviation",
            help="Here you are able to see the standard deviation of the results for all the models per sentence",
            format="%.2f%%",
        ),
        "Selected": st.column_config.Column(
            "Selected",
            help="Tick to select this row for OpenAI response evaluation",
        )
        if use_openai_response
        else None,
    },
)

percentage_column_config = {
    col: st.column_config.NumberColumn(
        format="%.2f%%", help=f"Percentage values for {col}"
    )
    for col in st.session_state.label_percentages_df.columns
    if col not in ["Std Deviation"]
}

percentage_column_config["Std Deviation"] = st.column_config.NumberColumn(
    format="%.2f%%", help="Standard deviation across models"
)

st.write("Percentages per Label for every Model")
st.dataframe(
    st.session_state.label_percentages_df,
    use_container_width=True,
    column_config=percentage_column_config,
    column_order=["Selected"]
    + [col for col in st.session_state.df.columns if col.startswith("Sentence")]
    + ["Label"]
    + ["Std Deviation"]
    + ["Aggregated Results"]
    + st.session_state.selected_model,
)

st.write("Performance per Model")
st.dataframe(
    st.session_state.overall_percentages_df,
    use_container_width=True,
    column_config=percentage_column_config,
    column_order=["Selected"]
    + [col for col in st.session_state.df.columns if col.startswith("Sentence")]
    + ["Label"]
    + ["Std Deviation"]
    + ["Aggregated Results"]
    + st.session_state.selected_model,
)


def get_changed_rows_df(original_df, edited_df):
    columns_to_check = [
        col
        for col in original_df.columns
        if col.startswith("Sentence") or col in ["Context", "Label"]
    ]

    # Check for changes between the original and edited DataFrame
    changes = original_df[columns_to_check] != edited_df[columns_to_check]

    # Identify rows where at least one column has changed
    rows_with_changes = changes.any(axis=1)

    # Extract the rows with changes from the original DataFrame
    changed_rows = original_df.loc[rows_with_changes].copy()

    # Add the modified columns from the edited DataFrame to the result
    for column in columns_to_check:
        changed_rows[f"modified_{column}"] = edited_df.loc[changed_rows.index, column]

    # Return only the relevant columns (original and modified ones)
    return changed_rows[
        [
            col
            for col in changed_rows.columns
            if "modified_" in col or col in columns_to_check
        ]
    ]


if "clicked" not in st.session_state:
    st.session_state.clicked = False


def click_button():
    st.session_state.clicked = True


if st.button("Proceed with Reevaluation", on_click=click_button):
    pass

if "openai_responses_list" not in st.session_state:
    st.session_state.openai_responses_list = []


def get_and_store_openai_responses(api_key, user_prompt_template, changes_df):
    responses_df = get_openai_responses(api_key, user_prompt_template, changes_df)
    st.session_state.openai_responses_list.append(
        {
            "prompt": user_prompt_template,
            "selected_rows": changes_df.copy(),
            "responses": responses_df.copy(),
        }
    )


if "clicked_openai" not in st.session_state:
    st.session_state.clicked_openai = False


def click_button_openai():
    st.session_state.clicked_openai = True


if use_openai_response:
    if st.button("Generate OpenAI Responses", on_click=click_button_openai):
        pass


def split_tuple_into_columns(input_value):
    try:
        # Check if the input is already a tuple
        if isinstance(input_value, tuple):
            return input_value
        # Safely evaluate the string into a tuple
        evaluated = eval(input_value)
        # Ensure the evaluated result is a tuple of strings
        if isinstance(evaluated, tuple):
            evaluated = tuple(str(item) for item in evaluated)
        return evaluated
    except Exception as e:
        st.error(f"Error evaluating input: {input_value} - {e}")
        return ("Error",)  # Return a placeholder in case of error


def validate_sentences(row):
    for col in sentence_columns:
        if pd.isna(row[col]) or not isinstance(row[col], str):
            return False
    return True


if st.session_state.clicked_openai:
    if use_openai_response:
        selected_rows = edited_df[edited_df["Selected"]]
        if not selected_rows.empty:
            with st.expander("OpenAI Prompt", expanded=True):
                st.write(user_prompt_template)
                st.write("Selected Rows for OpenAI Response:")
                st.dataframe(selected_rows, use_container_width=True)

            get_and_store_openai_responses(api_key, user_prompt_template, selected_rows)

            st.write("All Versions of OpenAI Prompts, Selected Rows, and Responses:")
            for idx, version in enumerate(
                st.session_state.openai_responses_list, start=1
            ):
                with st.expander(f"Version {idx}", expanded=False):
                    st.markdown("### Prompt")
                    st.text(version["prompt"])
                    st.markdown("### Selected Rows")
                    st.dataframe(version["selected_rows"], use_container_width=True)
                    st.markdown("### Responses")
                    st.dataframe(version["responses"], use_container_width=True)
        else:
            st.write("No rows selected for OpenAI response generation.")
    st.session_state.clicked_openai = False


if st.session_state.clicked:
    changes_df = get_changed_rows_df(st.session_state.df, edited_df)
    if not changes_df.empty:
        st.write("Changed rows:")

        # Run evaluation for multiple models
        model_predictions = sentence_eval(
            changes_df, selected_models=st.session_state.selected_model
        )

        # Initialize a list to store model data for merging
        dfs_to_merge = []

        for model_name, model_df in model_predictions.items():
            # Rename columns and clean the DataFrame
            model_df = model_df.rename(
                columns={
                    "instance": "Sentence",
                    "label": "Label",
                    "pred": f"{model_name} Prediction",
                }
            ).drop(["loss"], axis=1)

            model_df["Sentence"] = model_df["Sentence"].map(split_tuple_into_columns)

            max_tuple_length = model_df["Sentence"].map(len).max()
            for i in range(max_tuple_length):
                model_df[f"Sentence {i+1}"] = model_df["Sentence"].map(
                    lambda x: x[i] if i < len(x) else None
                )

            # Drop the original "Sentence" column if it's no longer needed
            model_df.drop(columns=["Sentence"], inplace=True)

            model_df[f"{model_name} Prediction"] = model_df[
                f"{model_name} Prediction"
            ].apply(lambda x: "{:.2f}%".format(x * 100))

            sentence_columns = [f"Sentence {i+1}" for i in range(max_tuple_length)]

            # Append to list for merging
            dfs_to_merge.append(model_df)

        # Merge all DataFrames on the 'Sentence' column
        # Refined merge logic for multiple sentences
        if dfs_to_merge:
            combined_df = dfs_to_merge[0]

            sentence_columns = [
                col for col in combined_df.columns if col.startswith("Sentence")
            ]

            for df in dfs_to_merge[1:]:
                combined_df = combined_df.merge(
                    df, on=sentence_columns + ["Label"], how="outer"
                )

                # Dynamically identify all columns named "Sentence 1" to "Sentence n"

            # Ensure consistent column order: Sentence parts first, Label, and then predictions
            columns_order = (
                sentence_columns
                + ["Label"]
                + [
                    col
                    for col in combined_df.columns
                    if col not in sentence_columns + ["Label"]
                ]
            )
            combined_df = combined_df[columns_order]

            # Filter rows to ensure all sentence parts are valid strings
            combined_df = combined_df[combined_df.apply(validate_sentences, axis=1)]

            # Display the combined DataFrame
            st.write("Combined Model Predictions:")
            st.dataframe(
                combined_df,
                hide_index=True,
                use_container_width=True,
            )

            st.session_state.evaluated_df = combined_df
    else:
        st.write("Please edit a sentence and its label to reevaluate.")
