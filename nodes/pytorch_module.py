"""
Module containing classes for classifications, detections ...

    - images
    - texts
    - raw data

a class per task such as : 

    - class ImageClassifier:
    - class ObjectDetector:
    - SentimentClassifier:

"""
import debugger
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# to-do : create custom transforms for grayscaling and other stuff

class Tester:
    test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    available_models = ["mobilenetv2", "resnet18", "resnet34", "resnet50"]
    """
    To init the class we need some infos that we'll receive through the server

    model_dict = {
        "classes_file" : [],         # list of strings
        "weights_path" : "",    # .pth file or "imagenet"
        "full_model_path" : "", # .pth file
        "model_arch_path" : "", # .py  file
        "device" : "",          # "cpu" or "cuda:0"
        "base_model": "resnet", # string or None
        "pretrained": True,     # bool (only for predefined models)
    }
    """
    def __init__(self, model_dict):

        self.device = model_dict["device"]
        self.classes = open(model_dict["classes_file"], "r").readlines()
        
        # available model case pretrained or not
        base_model = model_dict["base_model"]
        if base_model in self.available_models:
            pretrained = model_dict["pretrained"]
            print(f"Loading {base_model} (pretrained = {pretrained}) with {len(self.classes)} outputs")
        
            if base_model == "mobilenetv2":
                self.model = models.mobilenet_v2(pretrained=pretrained)
                self.model.classifier = nn.Sequential(
                    nn.Linear(
                        in_features = 1280, 
                        out_features = len(self.classes)
                    ), 
                    nn.Sigmoid()
                )

            if base_model == "resnet18":
                self.model = models.resnet18(pretrained=pretrained)
            if base_model == "resnet34":
                self.model = models.resnet34(pretrained=pretrained)
            if base_model == "resnet50":
                self.model = models.resnet50(pretrained=pretrained)

            self.model.fc = nn.Sequential(
                nn.Linear(
                    in_features = self.model.fc.in_features,
                    out_features = len(self.classes)
                ),
                nn.Sigmoid()
            )
        
        # custom model case from .py file
        model_arch_path = model_dict["model_arch_path"]
        if model_arch_path:
            # add file to path
            # import file with class name
            import importlib.util

            spec = importlib.util.spec_from_file_location("module.name", "tf_module.py")
            model_file = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_file)
            
            model = model_file.Model()
            if isinstance(model, torch.nn.Module):
                self.model = model
            else:
                raise 'Wrong model type'

        # avalable model + custom weights case
        # custom model   + custom weights case
        device = torch.device(model_dict["device"])
        weights_path = model_dict["weights_path"]
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=device))

        # full saved model case with weights included
        full_model_path = model_dict["full_model_path"]
        if full_model_path:
            self.model = torch.load(full_model_path, map_location=device)

        self.model.eval()

        print(self.model)
    
    @debugger.timeit
    def infer(self, data):
        return  self.model(data)

    def predict_path(self, images):

        for im_path in images:

            print("Predicting -> ", im_path)

            im = Image.open(im_path)

            im = self.test_transforms(im).unsqueeze(0)

            output = self.infer(im)
            #output = self.model(im)

            _, preds = torch.max(output, 1)

            print(output, preds)