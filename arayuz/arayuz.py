import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import subprocess
import os

class YOLOv7SegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HOŞ GELDİNİZ")
        self.root.geometry("800x600")  # Pencere boyutunu ayarlayabilirsiniz.

        self.image_path = None

        self.root.configure(bg="#f0f0f0")


        background_image = Image.open("arkaplan.jpg")
        background_image = ImageTk.PhotoImage(background_image)
        background_label = tk.Label(root, image=background_image)
        background_label.place(relwidth=1, relheight=1)

        # Create widgets
        self.label = tk.Label(root, text="DİŞ RESTORASYONLARI TESPİTİ", font=("Arial", 24, "bold"), bg="#f0f0f0", fg="#333")
        self.label.pack(pady=20)

        self.upload_button = tk.Button(root, text="3B PRT GÖRÜNTÜSÜ YÜKLE", command=self.upload_image, font=("Arial", 14, "bold"), bg="#4CAF50", fg="white", height=2, width=30)
        self.upload_button.pack(pady=10)

        self.process_button = tk.Button(root, text="RESTORASYONLARI TESPİT ET", command=self.process_image, font=("Arial", 14, "bold"), bg="#008CBA", fg="white", height=2, width=30, state=tk.DISABLED)
        self.process_button.pack(pady=10)

        self.image_label = tk.Label(root, bg="#f0f0f0")
        self.image_label.pack(pady=10)

        self.processed_image_label = tk.Label(root, bg="#f0f0f0")
        self.processed_image_label.pack(pady=10)

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if self.image_path:
            image = Image.open(self.image_path)
            image.thumbnail((640, 640))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

            # Resim seçildiğinde, resmi işleme butonunu aktifleştir
            self.process_button["state"] = tk.NORMAL

    def process_image(self):
        if self.image_path:
            output_image_path = os.path.splitext(self.image_path)[0] + "_output.png"

            subprocess.run([
                "python",
                f"{HOME}/yolov7/seg/segment/predict.py",
                "--weights", f"{HOME}/yolov7/seg/runs/train-seg/custom/weights/best.pt",
                "--conf", "0.50",
                "--source", self.image_path,
                "--name", output_image_path
            ])

            # YOLOv7 tarafından işlenmiş fotoğrafı göster
            output_image = Image.open(output_image_path + "/" + os.path.basename(self.image_path))
            output_image.thumbnail((640, 640))
            photo = ImageTk.PhotoImage(output_image)
            self.processed_image_label.config(image=photo)
            self.processed_image_label.image = photo

            # Dosyayı silerek kaydetme işlemi
            os.remove(output_image_path)

if __name__ == "__main__":
    HOME = os.path.expanduser("~")

    root = tk.Tk()
    app = YOLOv7SegmentationApp(root)
    root.mainloop()