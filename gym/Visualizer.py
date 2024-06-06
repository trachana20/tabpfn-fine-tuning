# from pathlib import Path
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# from tensorboardX import SummaryWriter


# class Visualizer:
#     def __init__(self, path):
#         self.path = path

#         self.base_path = Path(f"{self.path}/visualization").mkdir(
#             parents=True,
#             exist_ok=True,
#         )

#         self.summary_writers = {}
#         self.current_writer = None

#         print("")

#     def register_writer(self, writer_name):
#         assert (
#             writer_name not in self.summary_writers
#         ), "writer_name already exists in Visualizer"

#         current_path = self.base_path + writer_name
#         self.summary_writers[writer_name] = SummaryWriter(training_run_path)

#         pretty_dict_str = self.config_to_str(config)
#         self.writers[self.training_run_path].add_text(
#             "Configuration",
#             pretty_dict_str,
#             0,
#         )
