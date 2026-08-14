from deepface import DeepFace
import cv2
import os
import numpy as np
from collections import defaultdict

KNOWN_FACES_DIR = "known_faces"
THRESHOLD = 0.7  #Euclidean distance
FRAME_SKIP = 3   #Process every 3rd frame to reduce pressure of processing every frame
USE_COSINE = True  #Cosine similarity

print("Loading known faces...")

known_embeddings = []
known_names = []

#Precomputing embeddings for all saved pics of each person
if not os.path.exists(KNOWN_FACES_DIR):
    print(f"Directory '{KNOWN_FACES_DIR}' not found!")
    exit()

person_count = 0
for person_name in os.listdir(KNOWN_FACES_DIR):
    person_folder = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue
    
    person_count += 1
    face_count = 0
    
    for file_name in os.listdir(person_folder):
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(person_folder, file_name)
        try:
            rep = DeepFace.represent(
                img_path,
                model_name="Facenet",
                detector_backend="opencv",  #Use opencv backend for consistency
                enforce_detection=False
            )
            embedding = rep[0]["embedding"]
            known_embeddings.append(np.array(embedding))
            known_names.append(person_name)
            face_count += 1
        except Exception as e:
            print(f"Failed to process: {img_path} - {str(e)}")
    
    print(f"Loaded {face_count} images for {person_name}")

if len(known_embeddings) == 0:
    print("No known faces found! Please add images to the 'known_faces' directory.")
    exit()

print(f"\n✓ Finished loading {len(known_embeddings)} face(s) from {person_count} person(s)!")
print("Starting camera...\n")

#Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Could not open camera")
    exit()

#We use OpenCV Haarcascade for fast detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

frame_count = 0
face_labels = {}  # Used to reduce flickering

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def identify_face(face_embedding):
    """Identify face using precomputed embeddings"""
    if USE_COSINE:
        #Higher similarity = better match
        similarities = [cosine_similarity(face_embedding, known_emb) 
                       for known_emb in known_embeddings]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        #For cosine similarity, use 0.4-0.6 threshold (higher = stricter)
        COSINE_THRESHOLD = 0.5
        if best_score > COSINE_THRESHOLD:
            return known_names[best_idx], best_score
    else:
        #Lower distance for a better match
        distances = [np.linalg.norm(known_emb - face_embedding) 
                    for known_emb in known_embeddings]
        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]
        
        if best_distance < THRESHOLD:
            return known_names[best_idx], best_distance
    
    return "Unknown", None

print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    #Here we detect faces every frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Processes embeddings only every Nth frame
    if frame_count % FRAME_SKIP == 0:
        face_labels.clear()
        
        for idx, (x, y, w, h) in enumerate(faces):
            #Skip tiny detections
            if w < 60 or h < 60:
                face_labels[idx] = "Unknown"
                continue
            
            #Added padding for better recognition
            pad = 20
            y1, y2 = max(0, y-pad), min(frame.shape[0], y+h+pad)
            x1, x2 = max(0, x-pad), min(frame.shape[1], x+w+pad)
            face_crop = frame[y1:y2, x1:x2]

            #Here we get embedding for detected face
            try:
                rep = DeepFace.represent(
                    face_crop, 
                    model_name="Facenet",
                    detector_backend="opencv",  #Match the loading backend
                    enforce_detection=False
                )
                face_embedding = np.array(rep[0]["embedding"])
                
                name, score = identify_face(face_embedding)
                face_labels[idx] = name
                
            except Exception as e:
                face_labels[idx] = "Unknown"
                continue

    #Draws rectangles on detected faces and shows their names
    for idx, (x, y, w, h) in enumerate(faces):
        name = face_labels.get(idx, "Processing...")
        
        #Color based on recognition
        color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        #Background for text
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame, (x, y-35), (x + text_size[0], y), color, -1)
        
        cv2.putText(
            frame, name, (x, y-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

    #Displays face count
    cv2.putText(
        frame, f"Faces: {len(faces)}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    cv2.imshow("Face Recognition - Facenet", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✓ Camera closed. Goodbye!")