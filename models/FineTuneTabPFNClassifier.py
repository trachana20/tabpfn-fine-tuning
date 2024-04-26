from tabpfn.scripts.model_builder import load_model_only_inference
import os


class FineTuneTabPFNClassifier:
    def __init__(self, tabpfn_classifier, path):
        self.tabpfn_classifier = tabpfn_classifier
        self.path = path

        if os.pathexists(path):
            fine_tuned_model = load_model_only_inference(path)
            # just swap out the self.model of the tabpfn_classifier with thei newModel
            #
        self.tabpfn_classifier.model = fine_tuned_model
