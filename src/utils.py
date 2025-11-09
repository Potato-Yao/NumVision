"""
Utility functions for the digit recognition project.
"""
import os


def create_project_directories():
    """Create necessary project directories."""
    directories = ['models', 'data', 'tests', 'results']

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


if __name__ == "__main__":
    print("Utility functions loaded successfully!")
    create_project_directories()

