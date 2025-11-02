"""
Create sample test images for digit recognition.
This script generates simple digit images for testing predictions.
"""
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np


def create_simple_digit_image(digit, size=(280, 280), save_path=None):
    """
    Create a simple image with a digit.

    Args:
        digit: Digit to draw (0-9)
        size: Image size
        save_path: Path to save the image
    """
    # Create white background
    img = Image.new('L', size, 'white')
    draw = ImageDraw.Draw(img)

    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 200)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 200)
        except:
            font = ImageFont.load_default()

    # Get text size and center it
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)

    # Draw the digit in black
    draw.text(position, text, fill='black', font=font)

    if save_path:
        img.save(save_path)
        print(f"Created: {save_path}")

    return img


def create_noisy_digit_image(digit, size=(280, 280), save_path=None):
    """
    Create a digit image with some noise.

    Args:
        digit: Digit to draw (0-9)
        size: Image size
        save_path: Path to save the image
    """
    # Create base image
    img = create_simple_digit_image(digit, size)

    # Convert to numpy array
    img_array = np.array(img)

    # Add slight noise
    noise = np.random.normal(0, 10, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    # Convert back to image
    img = Image.fromarray(img_array)

    if save_path:
        img.save(save_path)
        print(f"Created: {save_path}")

    return img


def create_handwritten_style_digit(digit, size=(280, 280), save_path=None):
    """
    Create a more handwritten-looking digit.

    Args:
        digit: Digit to draw (0-9)
        size: Image size
        save_path: Path to save the image
    """
    img = Image.new('L', size, 'white')
    draw = ImageDraw.Draw(img)

    # Use a more casual font if available
    try:
        font = ImageFont.truetype("comic.ttf", 180)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 180)
        except:
            font = ImageFont.load_default()

    # Add some rotation for handwritten feel
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)

    # Draw with slight variations
    draw.text(position, text, fill='black', font=font)

    # Add slight blur effect by resizing
    img = img.resize((size[0]//2, size[1]//2), Image.Resampling.BILINEAR)
    img = img.resize(size, Image.Resampling.BILINEAR)

    if save_path:
        img.save(save_path)
        print(f"Created: {save_path}")

    return img


def main():
    """Generate sample test images."""
    # Create tests directory
    os.makedirs('tests', exist_ok=True)

    print("Generating sample test images...\n")

    # Generate simple digits
    print("Creating simple digits...")
    for digit in range(10):
        create_simple_digit_image(
            digit,
            save_path=f'tests/digit_{digit}_simple.png'
        )

    print("\nCreating noisy digits...")
    for digit in range(10):
        create_noisy_digit_image(
            digit,
            save_path=f'tests/digit_{digit}_noisy.png'
        )

    print("\nCreating handwritten-style digits...")
    for digit in range(10):
        create_handwritten_style_digit(
            digit,
            save_path=f'tests/digit_{digit}_handwritten.png'
        )

    print("\n" + "="*60)
    print("✅ Sample images created successfully!")
    print(f"   Location: tests/")
    print(f"   Total images: {30}")
    print("="*60)
    print("\nYou can test these images using:")
    print("  python main.py --mode predict --image tests/digit_5_simple.png")


if __name__ == "__main__":
    main()

