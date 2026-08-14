# Real-Time Face Recognition System

A real-time face recognition system built with Python, OpenCV, and DeepFace. It detects faces from a live webcam feed and matches them against a known-faces database using cosine similarity on Facenet embeddings.

## Features

- Real-time face detection via OpenCV Haar Cascade
- Face embeddings generated using DeepFace (Facenet model)
- Matching via cosine similarity against precomputed known-face embeddings
- Performance optimizations: frame-skipping, padded face crops, and filtering of low-confidence/undersized detections
- Robust camera initialization with fallback across capture backends and indices
- Live on-screen feedback: color-coded bounding boxes (green = matched, orange = unknown) and a running face count

## Tech Stack

- Python
- OpenCV
- DeepFace (Facenet)
- NumPy

## Project Structure

- `main.py` 
- `requirements.txt` — Python dependencies
- `known_faces/` — Reference photos (not tracked in git)
  - `person_name/photo1.jpg`, `photo2.jpg`, etc. — one subfolder per person
- `Tools/testpics.py` — Diagnostic script that checks images for corruption/size issues

## Setup

1. Clone the repository
2. Install dependencies
3. **Add reference photos**  
   Create a `known_faces/` folder in the project root. Inside it, create one subfolder per person, named after them, and add a few clear photos of their face.

   Example:
   - `known_faces/alice/photo1.jpg`
   - `known_faces/alice/photo2.jpg`
   - `known_faces/bob/photo1.jpg`
4. Run the program

Press `q` to quit the webcam window.

## Troubleshooting

If faces aren't being recognized, the most common cause is a problem with the reference photos themselves (corrupted files or images that are too small/low quality for DeepFace to process). Run the diagnostic script to check:

## Tools- testpics.py

This checks every image in `known_faces/` for readability, file corruption, and minimum size requirements, and reports which files are causing issues.

## Notes

- The `known_faces/` folder is excluded from version control via `.gitignore` since it contains personal photos.
- DeepFace automatically downloads model weights on first run (stored in `.deepface/`, also excluded from git).
