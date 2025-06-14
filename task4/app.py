from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import base64
from io import BytesIO
from pix2pix import UNetGenerator

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = UNetGenerator().to(device)
generator.load_state_dict(torch.load('generator.pth', map_location=device))
generator.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        img_data = base64.b64decode(data.split(',')[1])
        img = Image.open(BytesIO(img_data)).convert('RGB')
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Generate output
        with torch.no_grad():
            output = generator(img_tensor)
        output = (output.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
        output_img = transforms.ToPILImage()(output)
        
        # Convert output to base64
        buffered = BytesIO()
        output_img.save(buffered, format="PNG")
        output_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
        
        return jsonify({'output_image': output_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)