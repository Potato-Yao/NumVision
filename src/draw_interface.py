"""
Simple drawing interface for testing digit recognition.
Allows users to draw digits with their mouse and get real-time predictions.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw
import numpy as np
import os


class DrawingApp:
    """Simple drawing application for digit recognition."""

    def __init__(self, root, model=None):
        """
        Initialize the drawing application.

        Args:
            root: Tkinter root window
            model: Pre-loaded Keras model (optional)
        """
        self.root = root
        self.root.title("NumVision - Draw a Digit")
        self.model = model

        # Canvas settings
        self.canvas_size = 280
        self.brush_size = 20

        # Create PIL image for drawing
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)

        self.setup_ui()

        # Drawing state
        self.last_x = None
        self.last_y = None

    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title = ttk.Label(main_frame, text="Draw a Digit (0-9)",
                         font=('Arial', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=2, pady=10)

        # Canvas
        self.canvas = tk.Canvas(main_frame, width=self.canvas_size,
                               height=self.canvas_size, bg='white',
                               cursor='cross')
        self.canvas.grid(row=1, column=0, columnspan=2, pady=10)

        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        # Predict button
        self.predict_btn = ttk.Button(button_frame, text="Predict",
                                      command=self.predict)
        self.predict_btn.pack(side=tk.LEFT, padx=5)

        # Clear button
        clear_btn = ttk.Button(button_frame, text="Clear",
                              command=self.clear_canvas)
        clear_btn.pack(side=tk.LEFT, padx=5)

        # Save button
        save_btn = ttk.Button(button_frame, text="Save Image",
                             command=self.save_image)
        save_btn.pack(side=tk.LEFT, padx=5)

        # Result frame
        result_frame = ttk.LabelFrame(main_frame, text="Prediction Result",
                                     padding="10")
        result_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        # Result label
        self.result_label = ttk.Label(result_frame,
                                     text="Draw a digit and click 'Predict'",
                                     font=('Arial', 14))
        self.result_label.pack()

        # Confidence label
        self.confidence_label = ttk.Label(result_frame, text="",
                                         font=('Arial', 10))
        self.confidence_label.pack()

        # Instructions
        instructions = ttk.Label(main_frame,
                                text="Draw a digit using your mouse",
                                font=('Arial', 9, 'italic'))
        instructions.grid(row=4, column=0, columnspan=2, pady=5)

    def start_draw(self, event):
        """Start drawing."""
        self.last_x = event.x
        self.last_y = event.y

    def draw_line(self, event):
        """Draw line on canvas."""
        if self.last_x and self.last_y:
            # Draw on canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                   width=self.brush_size, fill='black',
                                   capstyle=tk.ROUND, smooth=True)

            # Draw on PIL image
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                          fill='black', width=self.brush_size)

        self.last_x = event.x
        self.last_y = event.y

    def stop_draw(self, event):
        """Stop drawing."""
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        """Clear the canvas."""
        self.canvas.delete('all')
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit and click 'Predict'")
        self.confidence_label.config(text="")

    def preprocess_image(self):
        """Preprocess the drawn image for prediction."""
        # Resize to 28x28
        img = self.image.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert to numpy array
        img_array = np.array(img)

        # Invert (MNIST has white digits on black background)
        img_array = 255 - img_array

        # Normalize
        img_array = img_array.astype('float32') / 255.0

        # Add dimensions
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self):
        """Make prediction on the drawn digit."""
        if self.model is None:
            messagebox.showerror("Error",
                               "No model loaded! Please train a model first.")
            return

        # Preprocess image
        processed_img = self.preprocess_image()

        # Make prediction
        predictions = self.model.predict(processed_img, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = predictions[0][predicted_digit]

        # Update result
        self.result_label.config(text=f"Predicted Digit: {predicted_digit}",
                                font=('Arial', 20, 'bold'))
        self.confidence_label.config(text=f"Confidence: {confidence*100:.2f}%")

        # Show top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_text = "Top 3: " + ", ".join([f"{i}({predictions[0][i]*100:.1f}%)"
                                            for i in top_3_idx])
        self.confidence_label.config(text=f"Confidence: {confidence*100:.2f}%\n{top_3_text}")

    def save_image(self):
        """Save the drawn image."""
        os.makedirs('tests', exist_ok=True)

        # Find next available filename
        counter = 1
        while os.path.exists(f'tests/drawn_digit_{counter}.png'):
            counter += 1

        filename = f'tests/drawn_digit_{counter}.png'

        # Save original size
        self.image.save(filename)

        # Also save preprocessed version
        processed = self.preprocess_image()
        processed_img = (processed[0] * 255).astype(np.uint8)
        Image.fromarray(processed_img.squeeze()).save(
            f'tests/drawn_digit_{counter}_processed.png'
        )

        messagebox.showinfo("Success", f"Image saved as {filename}")


def launch_drawing_app(model=None):
    """
    Launch the drawing application.

    Args:
        model: Pre-loaded Keras model (optional)
    """
    root = tk.Tk()
    app = DrawingApp(root, model)
    root.mainloop()
