from flask import Flask, render_template, request, send_file
import os
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models import vgg19

device = "cuda" if torch.cuda.is_available() else "cpu"
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_img(path):
    image = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize(512),
        T.CenterCrop(512),
        T.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    return torch.bmm(features, features.transpose(1, 2)) / (c * h * w)

def get_features(model, x, content_layers, style_layers):
    features = {}
    for name, layer in model._modules.items():
        x = layer(x)
        if name in content_layers or name in style_layers:
            features[name] = x
    return features

@app.route("/", methods=["GET"])
def index():
    return open("index.html").read()

@app.route("/upload", methods=["POST"])
def upload():
    content_file = request.files["content"]
    style_file = request.files["style"]

    content_path = os.path.join(UPLOAD_FOLDER, "content.jpg")
    style_path = os.path.join(UPLOAD_FOLDER, "style.jpg")
    content_file.save(content_path)
    style_file.save(style_path)

    content = load_img(content_path)
    style = load_img(style_path)

    model = vgg19(weights="IMAGENET1K_V1").features.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    content_layers = {'21'}
    style_layers = {'0','5','10','19','28'}

    content_feat = get_features(model, content, content_layers, style_layers)['21']
    style_feats = get_features(model, style, content_layers, style_layers)
    style_grams = {l: gram_matrix(style_feats[l]) for l in style_layers}

    generated = content.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([generated], lr=0.02)

    for step in range(300):
        gen_feats = get_features(model, generated, content_layers, style_layers)
        content_loss = torch.nn.functional.mse_loss(gen_feats['21'], content_feat)
        style_loss = sum(
            torch.nn.functional.mse_loss(gram_matrix(gen_feats[l]), style_grams[l]) for l in style_layers
        )
        loss = content_loss + 1e5 * style_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    result = generated.squeeze().cpu().clamp(0, 1)
    result_img = T.ToPILImage()(result)
    output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
    result_img.save(output_path)
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True)
