import torch.nn as nn
import torch
import os
import loralib as lora

class MockTabPFN(nn.Module):
    def __init__(self, tabpfn_classifier):
        super(MockTabPFN, self).__init__()
        self.linear = nn.Linear(100, 10, False)
        self.tabpfn_classifier = tabpfn_classifier

        self.lora_finetuning(self.tabpfn_classifier, False)
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

    def lora_finetuning(self, tabpfn, finetune):

        '''
        Finetuning Technique #1: LORA
        
        Replaces with lora layers to TabPFNClassifier wherever possible.

        Args:
            tabpfn: original TabPFNClassifier instance

        '''
        
        model = tabpfn.model[2]
        
        if not finetune:
            self.model = model
        
        else:
            for _, layer in enumerate(model.transformer_encoder.layers):
                layer.linear1 = lora.Linear(layer.linear1.in_features, layer.linear1.out_features, r=16)
                layer.linear2 = lora.Linear(layer.linear2.in_features, layer.linear2.out_features, r=16)
            
            # model.encoder = lora.Linear(model.encoder.in_features, model.encoder.out_features, r=16)  ## Assigning LORA to this gives NaN validation output
            model.y_encoder = lora.Linear(model.y_encoder.in_features, model.y_encoder.out_features, r=16)
            model.decoder[0] = lora.Linear(model.decoder[0].in_features, model.decoder[0].out_features, r=16)
            # model.decoder[2] = lora.Linear(model.decoder[2].in_features, model.decoder[2].out_features, r=16) ## Assigning LORA to this gives NaN validation output

            self.model = model
