import torch
import torch.nn as nn
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = CNN()
model.load_state_dict(torch.load("cnn_mnist.pth", map_location=torch.device('cpu')))
model.eval()

def predict(img_array):
    img_tensor = torch.tensor(img_array).view(1, 1, 28, 28).float()
    img_tensor = (img_tensor - 0.1307) / 0.3081 
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        return torch.argmax(prob).item()


def get_digits(img):
    arr = np.array(img)

    cols = np.any(arr > 0, axis=0)
    if not np.any(cols): return []

    boundaries = np.where(diff(cols.astype(int)) != 0)[0]

    from scipy.ndimage import label, find_objects
    mask = arr > 0
    labeled, n = label(mask)
    objects = find_objects(labeled)
    
    return sorted(objects, key=lambda x: x[1].start)

def check():
    arr = np.array(image)
    from scipy.ndimage import label, find_objects
    labeled, n = label(arr > 0)
    objects = find_objects(labeled)
    
    if not objects:
        label_res.config(text="Draw something!")
        return

    sorted_objects = sorted(objects, key=lambda x: x[1].start)
    
    full_number = ""
    for obj in sorted_objects:
        digit_crop = image.crop((obj[1].start, obj[0].start, obj[1].stop, obj[0].stop))
        
        size = max(digit_crop.size)
        new_img = Image.new("L", (size + 20, size + 20), 0)
        new_img.paste(digit_crop, (10, 10))
        img_final = np.array(new_img.resize((28, 28))) / 255.0
        
        digit = predict(img_final)
        full_number += str(digit)
    
    label_res.config(text=f"Number: {full_number}")

root = tk.Tk()
canvas_width = 600
canvas_height = 200
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='black')
canvas.pack()
image = Image.new("L", (canvas_width, canvas_height), 0)
draw = ImageDraw.Draw(image)

def paint(event):
    x, y = event.x, event.y
    canvas.create_oval(x-8, y-8, x+8, y+8, fill="white", outline="white")
    draw.ellipse([x-8, y-8, x+8, y+8], fill=255)

canvas.bind("<B1-Motion>", paint)
tk.Button(root, text="Predict Number", command=check).pack()
tk.Button(root, text="Clear", command=lambda: (canvas.delete("all"), draw.rectangle([0,0,600,200], fill=0))).pack()
label_res = tk.Label(root, text="Draw a number (e.g. 123)", font=("Arial", 20))
label_res.pack()
root.mainloop()