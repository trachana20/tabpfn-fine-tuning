from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter
import json
import pandas as pd


class Visualizer:
    def __init__(self, path):
        self.path = path

        self.base_path = f"{self.path}visualization"
        Path(self.base_path).mkdir(parents=True, exist_ok=True)

        self.summary_writers = {}
        self.current_path = f"{self.path}"

    def register_writer(self, writer_name, config):
        temp = self.current_path
        self.current_path = f"{self.base_path}/{writer_name}"

        if temp != self.current_path and temp is not None:
            self.summary_writers[temp].close()

        assert (
            self.current_path not in self.summary_writers
        ), "writer_name already exists in Visualizer"
        self.summary_writers[self.current_path] = SummaryWriter(self.current_path)

        pretty_dict_str = self.config_to_str(config)
        self.summary_writers[self.current_path].add_text(
            "Configuration",
            pretty_dict_str,
            0,
        )

    def update_scalar_value(self, value_name, value, epoch):
        # Check if value is serializable using the helper function
        if not self._is_serializable(value):
            raise ValueError("Provided value is not serializable")
        # Log the value to TensorBoard
        self.summary_writers[self.current_path].add_scalar(value_name, value, epoch)

    # ----- ----- -----  U T I L I T Y  ----- ----- -----
    def config_to_str(self, config, level=0):
        """Convert a configuration dictionary to a formatted Markdown string."""
        lines = []
        indent = "    " * level  # Adjusted indenting for visibility in Markdown
        for key, value in config.items():
            if isinstance(value, dict):
                # For nested dictionary, add the key and then format the nested dict
                lines.append(f"{indent}- **{key}**:")
                nested = self.config_to_str(value, level + 1)
                nested_lines = nested.split("\n")
                lines.extend(nested_lines)
            else:
                # For simple key-value pairs, format directly
                lines.append(f"{indent}- **{key}**: {value}")
        return "\n".join(lines)

    def _is_serializable(self, value):
        """Helper function to check if a value is serializable."""
        try:
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False

    def save_training_logs_as_csv(self):
        self.close()
        model_log_dir = f"{self.current_path}"
        csv_file_path = f"{self.current_path}_training_logs.csv"

        # Create an instance of EventAccumulator
        event_acc = EventAccumulator(model_log_dir)
        event_acc.Reload()  # Loads the log data

        # Organize scalar data
        organized_data = {}
        for tag in event_acc.Tags()["scalars"]:
            events = event_acc.Scalars(tag)
            for e in events:
                if e.step not in organized_data:
                    organized_data[e.step] = {"Step": e.step}
                organized_data[e.step][tag] = e.value

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(organized_data, orient="index").reset_index(
            drop=True,
        )

        # Save to CSV
        df.to_csv(csv_file_path, index=False)
        
    def save_results(self):
        # Check if evaluation_data is not empty
        if self.evaluation_data:
            # Create the directory if it doesn't exist

            os.makedirs(os.path.dirname(self.results_path), exist_ok=True)

            file_name = self.results_path + "fit_predict.json"
            # Open the file in write mode and save the data as JSON
            with open(file_name, "w") as file:
                json.dump(self.evaluation_data, file)
        else:
            print("No evaluation data to save.")

    def close(self):
        # Close all the writers when done
        for writer in self.summary_writers.values():
            writer.close()
