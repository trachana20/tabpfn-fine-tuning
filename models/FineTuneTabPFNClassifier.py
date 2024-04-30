# from tabpfn.scripts.model_builder import load_model_only_inference
# import os
import torch.nn as nn
import torch
from models.MockTabPFN import MockTabPFN


class FineTuneTabPFNClassifier:
    def __init__(self, tabpfn_classifier, path):
        self.tabpfn_classifier = tabpfn_classifier
        self.path = path

        # if os.pathexists(path):
        #     fine_tuned_model = load_model_only_inference(path)
        # just swap out the self.model of the tabpfn_classifier with the new model
        #
        fine_tuned_model = MockTabPFN()

        self.tabpfn_classifier.model = (None, None, fine_tuned_model)
        self.tabpfn_classifier.model_name = "FineTuneTabPFN"

    def fit(self, X, y):
        return self.tabpfn_classifier.fit(X=X, y=y)

    def predict(self, X):
        return self.tabpfn_classifier.predict(X=X)

    def predict_proba(self, X):
        return self.tabpfn_classifier.predict_proba(X=X)
