import torch.nn as nn
import torch
import os
# Define the RAG components
# tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
# retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="custom", passages_path="data/passages.tsv")
# rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

class MockTabPFN(nn.Module):
    def __init__(self, tabpfn_classifier):
        super(MockTabPFN, self).__init__()
        self.linear = nn.Linear(100, 10, False)
        # Initialize the weights with random values
        torch.nn.init.normal_(self.linear.weight, mean=0, std=10)

        # store the weight in model_weights/MockTabPFN.pth
        weights_path = "model_weights/MockTabPFN.pth"
        self.save_weights(weights_path)

    def forward(self, X, single_eval_pos):
        out = self.linear(X[1])
        out = out[single_eval_pos:]
        return out

    def save_weights(self, file_path):
        # Ensure the path exists
        dir_path = os.path.join(*file_path.split("/")[:-1])
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.linear.weight, file_path)

# class FineTuneTabPFNClassifierWithRAG(FineTuneTabPFNClassifier):
#     def __init__(self, tabpfn_classifier, weights_path, rag_model, tokenizer):
#         super().__init__(tabpfn_classifier, weights_path)
#         self.rag_model = rag_model
#         self.tokenizer = tokenizer

#     def fit(self, X, y):
#         rag_inputs = self.tokenizer(X, return_tensors="pt", truncation=True, padding=True)
#         rag_outputs = self.rag_model(**rag_inputs)
#         augmented_X = rag_outputs.logits.argmax(dim=-1).tolist()
#         return self.tabpfn_classifier.fit(X=augmented_X, y=y)

#     def predict(self, X):
#         rag_inputs = self.tokenizer(X, return_tensors="pt", truncation=True, padding=True)
#         rag_outputs = self.rag_model(**rag_inputs)
#         augmented_X = rag_outputs.logits.argmax(dim=-1).tolist()
#         return self.tabpfn_classifier.predict(X=augmented_X)

#     def predict_proba(self, X):
#         rag_inputs = self.tokenizer(X, return_tensors="pt", truncation=True, padding=True)
#         rag_outputs = self.rag_model(**rag_inputs)
#         augmented_X = rag_outputs.logits.argmax(dim=-1).tolist()
#         return self.tabpfn_classifier.predict_proba(X=augmented_X)
