from flask import Flask, render_template, request
from PIL import Image
import torch
from torchvision import transforms
import io
import torch.nn as nn

app = Flask(__name__)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        ])

        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 36)
        )

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(x.size(0), -1)
        output = self.fc_layers(x)

        return output, x

model = CNN()
weights_path = r"C:\Users\Kishore\Box\Fall 2023\Intro to ML\Bonus\Model Deployment\cnn_model_weights.pth"

model.load_state_dict(torch.load(weights_path))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1758], std=[0.333])
])

@app.route('/')
def upload():
    return render_template("webpage.html")

@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template("webpage.html", message='No file part')

        f = request.files['file']

        if f.filename == '':
            return render_template("webpage.html", message='No selected file')

        #Reading the uploaded image
        image_stream = f.read()
        image = Image.open(io.BytesIO(image_stream))
        image_tensor = transform(image).unsqueeze(0)

        # Predicting the output from the input image
        with torch.no_grad():
            output,_ = model(image_tensor)
            _, predicted = torch.max(output.data, 1)

        class_mapping = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
        20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
        30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
        }       

        return render_template("webpage.html", message=f"Model Prediction: {class_mapping[predicted.item()]}")

if __name__ == '__main__':
    app.run(debug=True)