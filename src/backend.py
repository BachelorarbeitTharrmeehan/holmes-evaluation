import pandas as pd
from utils import data_loading
from defs.control_task_types import CONTROL_TASK_TYPES
from defs.probe_task_types import PROBE_TASK_TYPES
from torch.optim import Adam
from model.probing_model import LinearProbingModel
from utils.data_loading import load_dataset
from pytorch_lightning import Trainer
import glob


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


def Backend(
    selected_models=["microsoft/deberta-v3-base"],
    probing_task="blimp-determiner_noun_agreement_with_adj_irregular_2",
):
    all_model_predictions = {}

    probe_frame = data_loading.load_probe_file(
        f"./data/holmes/{probing_task}/modified_samples.csv",
        CONTROL_TASK_TYPES.NONE,
    )

    for model in selected_models:
        # Load model and dataset
        base_model = data_loading.load_model(model, CONTROL_TASK_TYPES.NONE, "full")
        probing_frames = data_loading.load_folds(
            probe_frame=probe_frame,
            base_model=base_model,
            probe_task_type=PROBE_TASK_TYPES.SENTENCE,
            encoding="full",
            encoding_batch_size=10,
        )
        loaded_probing_frames = data_loading.load_probing_frames(probing_frames, "full")
        input_dim = loaded_probing_frames[0]["test"].iloc[0]["inputs_encoded"].shape[-1]

        # Define configuration
        cfg = {
            "learning_rate": 0.001,
            "num_labels": 2,
            "input_dim": input_dim,
            "batch_size": 16,
            "optimizer": Adam,
            "hidden_dim": 0,
            "dropout": 0.2,
            "warmup_rate": 0.1,
            "num_hidden_layers": 0,
        }

        all_predictions = []

        # Iterate over seeds from 1 to 5
        for seed in range(0, 5):
            cfg["seed"] = seed  # Update seed in configuration
            model_renamed = model.replace("/", "__")
            path_pattern = f"./results/holmes/{probing_task}/{model_renamed}/full/NONE/**/{seed}/0/done/*.ckpt"
            checkpoint_files = glob.glob(path_pattern)

            # Load model checkpoint for each seed
            probing_model = LinearProbingModel.load_from_checkpoint(
                checkpoint_path=checkpoint_files[0],
                hyperparameter=cfg,
            )
            test_data_probing_frame = loaded_probing_frames[0]["test"]
            test_dataset = load_dataset(test_data_probing_frame)
            custom_dataloader = probing_model.get_test_dataloader(
                test_dataset, 300, shuffle=False
            )
            trainer = Trainer(accelerator="auto", devices="auto", precision="32")
            trainer.test(probing_model, dataloaders=[custom_dataloader])

            # Collect predictions for this seed
            predictions = [
                (
                    instance_input,
                    modify_prediction({"label": instance_label, "pred": pred}),
                    instance_label,
                    loss,
                )
                for instance_input, instance_label, pred, loss in zip(
                    test_dataset.inputs,
                    test_dataset.labels,
                    probing_model.test_preds,
                    probing_model.test_losses,
                )
            ]
            all_predictions.append(
                pd.DataFrame(predictions, columns=["instance", "pred", "label", "loss"])
            )

        # Average predictions across all seeds
        combined_predictions = pd.concat(all_predictions, axis=0)
        average_predictions = (
            combined_predictions.groupby("instance").mean().reset_index()
        )
        all_model_predictions[model] = average_predictions

        # Save the averaged predictions
        model_name_safe = model.replace("/", "__")
        average_predictions.to_csv(
            f"./results/holmes/{probing_task}/{model_name_safe}/averaged_predictions.csv",
            index=False,
        )

    return all_model_predictions
