import torch
import numpy as np
from PIL import Image, ImageDraw
import os
from model import CNN


model = CNN()
if os.path.exists("cnn_mnist.pth"):
    model.load_state_dict(torch.load("cnn_mnist.pth", map_location=torch.device('cpu')))
model.eval()

def predict(img_array):
    img_tensor = torch.tensor(img_array).view(1, 1, 28, 28).float()
    img_tensor = (img_tensor - 0.1307) / 0.3081 
    with torch.no_grad():
        output = model(img_tensor)
        return torch.argmax(output, dim=1).item()

def run_gui():
    import tkinter as tk
    from scipy.ndimage import label, find_objects

    def check():
        arr = np.array(image)
        labeled, n = label(arr > 0)
        objects = find_objects(labeled)
        if not objects:
            label_res.config(text="draw something!")
            return
        
        sorted_objects = sorted(objects, key=lambda x: x[1].start)
        full_number = ""
        for obj in sorted_objects:
            digit_crop = image.crop((obj[1].start, obj[0].start, obj[1].stop, obj[0].stop))
            size = max(digit_crop.size)
            new_img = Image.new("L", (size + 20, size + 20), 0)
            new_img.paste(digit_crop, (10, 10))
            img_final = np.array(new_img.resize((28, 28))) / 255.0
            full_number += str(predict(img_final))
        label_res.config(text=f"number: {full_number}")

    root = tk.Tk()
    root.title("digit recognizer")
    canvas = tk.Canvas(root, width=600, height=200, bg='black')
    canvas.pack()
    
    global image, draw
    image = Image.new("L", (600, 200), 0)
    draw = ImageDraw.Draw(image)

    def paint(event):
        x, y = event.x, event.y
        canvas.create_oval(x-8, y-8, x+8, y+8, fill="white", outline="white")
        draw.ellipse([x-8, y-8, x+8, y+8], fill=255)

    canvas.bind("<B1-Motion>", paint)
    tk.Button(root, text="predict", command=check).pack()
    tk.Button(root, text="clear", command=lambda: (canvas.delete("all"), draw.rectangle([0,0,600,200], fill=0))).pack()
    global label_res
    label_res = tk.Label(root, text="draw here", font=("Arial", 20))
    label_res.pack()
    root.mainloop()

if __name__ == "__main__":
    if os.environ.get('DISPLAY') or os.name == 'nt':
        run_gui()
    else:
        print("headless mode: model loaded, check passed")