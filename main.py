
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import os
import __main__
import io
import base64
from pydantic import BaseModel

def data_transforms(img):
    transformations = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor()])
    img_tensor = transformations(img)
    return img_tensor

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))


class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(classes))
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)



app = FastAPI()

classes = ['cardboard', 'glass', 'metal', 'paper', 'pe-hd', 'pet', 'plastic', 'pp', 'ps', 'trash']

setattr(__main__, "ResNet", ResNet)
model = torch.load("model_resnet50.pt", map_location=torch.device('cpu'))


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    device = get_default_device()
    img_tensor = data_transforms(image)
    # Convert to a batch of 1
    xb = to_device(img_tensor.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    predicted_label = classes[preds[0].item()]
    return {"class": predicted_label}




class ImageRequest(BaseModel):
    image: str

@app.post("/encode_base64")
async def process_image(request: ImageRequest):
    # Decode the base64-encoded image data
    encoded_image = request.image
    binary_image = base64.b64decode(encoded_image)

    # Load the image data into a PyTorch tensor
    image_bytes = io.BytesIO(binary_image)
    image = transforms.ToTensor()(PIL.Image.open(image_bytes))

    # Do something with the PyTorch tensor
    # ...

    # Return a response
    return {"message": "Image processed successfully"}


@app.post("/encode_predict")
async def process_image(request: ImageRequest):
    # Decode the base64-encoded image data
    encoded_image = request.image
    binary_image = base64.b64decode(encoded_image)

    # Load the image data into a PyTorch tensor
    image_bytes = io.BytesIO(binary_image)
    image = transforms.ToTensor()(PIL.Image.open(image_bytes))

    device = get_default_device()
    img_tensor = data_transforms(image)
    # Convert to a batch of 1
    xb = to_device(img_tensor.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    predicted_label = classes[preds[0].item()]
    return {"class": predicted_label}




