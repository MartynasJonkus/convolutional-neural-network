import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from PIL import Image
from flask import Flask, request, jsonify
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class Convolutional_Neural_Network(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        channels = [3, 32, 64, 128, 256]
        self.features = nn.Sequential()

        for i in range(len(channels)-1):
            self.features.add_module(f'conv_block{i+1}', nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Convolutional_Neural_Network(num_classes=3).to(device)
model.load_state_dict(torch.load('trained_model.pth', map_location=device))
model.eval()

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

class_names = ['jellyfish', 'owl', 'pizza']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        with torch.no_grad():
            outputs = model(tensor)
        
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probabilities, 1)
        
        return jsonify({
            'class': class_names[pred_idx.item()],
            'confidence': conf.item(),
            'probabilities': {name: float(prob) for name, prob in zip(class_names, probabilities[0].tolist())}
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)