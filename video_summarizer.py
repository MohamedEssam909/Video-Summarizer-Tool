import cv2
import os
import numpy as np
from moviepy import VideoFileClip ,concatenate_videoclips
from pydub import AudioSegment
import moviepy.config as mpy_config
import sys
import statistics


config_path = "deploy.prototxt.txt"
model_path = "mobilenet_iter_73000.caffemodel"




if getattr(sys, 'frozen', False):
    ffmpeg_path = os.path.join(sys._MEIPASS, 'ffmpeg.exe') 
else:
    ffmpeg_path = 'ffmpeg.exe'  

os.environ["FFMPEG_BINARY"] = ffmpeg_path
mpy_config.FFMPEG_BINARY = ffmpeg_path




def extract_frames(video_path, output_folder, fps=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"video fps is! : {video_fps}")
    frame_interval = int(video_fps / fps)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:  #boolean flag to indicate frame is captured or not 
            break

        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames to {output_folder}.")



'''
 Computes a color histogram for a given video frame. 
 This histogram is a numerical representation of the color distribution, 
 useful for comparing visual similarity between frames.
'''

def calculate_histogram(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()





def detect_scene_changes(frames_folder, threshold=0.7):
    frames = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    if not frames:
        print("No frames found in folder.")
        return []

    scene_changes = [frames[0]]
    last_hist = None

    for frame_name in frames:
        frame_path = os.path.join(frames_folder, frame_name)
        frame = cv2.imread(frame_path)
        hist = calculate_histogram(frame)

        if last_hist is not None:
            diff = cv2.compareHist(hist, last_hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > threshold:
                scene_changes.append(frame_name)

        last_hist = hist

    print(f"Detected {len(scene_changes)} scene changes.")
    return scene_changes





def score_scene(scene_video_path):
    face_score = compute_face_score(scene_video_path)
    motion_score = compute_motion_score(scene_video_path)
    audio_score = compute_audio_score(scene_video_path)
    object_score = compute_object_score(scene_video_path)
    total_score = (
        face_score * 0.25 +
        motion_score * 0.25 +
        audio_score * 0.25 +
        object_score * 0.25
    )
    return total_score




def frame_to_seconds(frame_index, fps):
    return frame_index / fps




def split_video_by_scenes(video_path, frames_folder, scene_frames, output_folder, sampled_fps=1):
    os.makedirs(output_folder, exist_ok=True)
    video = VideoFileClip(video_path)
    duration = video.duration

    all_frames = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    total_frames = len(all_frames)

    # Actual spacing between each sampled frame
    frame_spacing_seconds = duration / total_frames

    # Convert scene frame names to indices
    scene_indices = [int(f.split("_")[1].split(".")[0]) for f in scene_frames]
    scene_indices.append(total_frames - 1)  # ensure last segment gets included

    # Convert frame indices to actual video times (in seconds)
    scene_times = [frame_to_seconds(i, frame_spacing_seconds) for i in scene_indices]

    for i in range(len(scene_times) - 1):
        start = scene_times[i]
        end = scene_times[i + 1]

    min_scene_duration = 2  # Minimum duration for scenes (in seconds)
    final_scene_times = []
    last_start = scene_times[0]

    for i in range(1, len(scene_times)):
        # If the current scene is too close to the last one, merge them
        if scene_times[i] - last_start < min_scene_duration:
            continue  # Skip adding a new scene here; continue with the current scene
        # Otherwise, store the current scene range
        final_scene_times.append((last_start, scene_times[i]))
        last_start = scene_times[i]

    # Ensure the last scene gets added
    final_scene_times.append((last_start, scene_times[-1]))
    scene_durations = [end - start for (start, end) in final_scene_times if end > start]
    avg_duration = sum(scene_durations) / len(scene_durations)
    print("here->",avg_duration)
    std_dev = statistics.stdev(scene_durations)

    for i, (start, end) in enumerate(final_scene_times):
        
        if start >= end or end > duration:
            continue

        if end-start<max(0.5, avg_duration - std_dev) or end-start>(avg_duration+std_dev):
            continue

        scene_clip = video.subclipped(start, end)
        temp_audio = os.path.join(output_folder, f"scene_{i:02d}_TEMP_AUDIO.mp3")
        output_video = os.path.join(output_folder, f"scene_{i:02d}.mp4")

        try:
            scene_clip.write_videofile(
                output_video,
                codec="libx264",
                audio=True,
                temp_audiofile=temp_audio,
                remove_temp=True,
                threads=4,
                logger=None
            )
            print(f" Generated: {output_video}")
        except Exception as e:
            print(f" Error writing scene {i}: {e}")

        if os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
                print(f" Removed {temp_audio}")
            except Exception as e:
                print(f" Could not remove temp audio file: {e}")      




# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

net = cv2.dnn.readNetFromCaffe(config_path, model_path)


class_names = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor"
]



def detect_objects(frame):
    # Get frame dimensions
    height, width = frame.shape[:2]

    # Prepare the image for input to the model
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), True)

    net.setInput(blob)
    detections = net.forward()

    # List to store detected objects
    detected_objects = []

    # Loop through detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Only keep detections with a confidence greater than 50%
            class_id = int(detections[0, 0, i, 1])
            detected_objects.append(class_names[class_id])

    return detected_objects



def compute_object_score(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    object_count_sum = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_objects = detect_objects(frame)
        object_count_sum += len(detected_objects)
        frame_count += 1

    cap.release()
    return object_count_sum / frame_count if frame_count > 0 else 0



def compute_audio_score(video_path):
    # Extract audio from video and load it with pydub
    audio_path = "temp_audio.wav"
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

    # Load audio file and analyze volume
    audio = AudioSegment.from_wav(audio_path)
    loudness = np.mean(np.array(audio.get_array_of_samples()))  # Average loudness (volume)
    
    os.remove(audio_path)  # Clean up temporary audio file

    # A threshold to determine "loud" audio (speech, music, etc.)
    return loudness

def compute_motion_score(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return 0  
    
    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        print("Failed to read the first frame.")
        return 0  
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    motion_score = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute optical flow (this method uses Farneback method for dense flow)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Calculate motion magnitude
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_score += np.sum(magnitude)  # Sum of all motion magnitudes

        prev_gray = curr_gray

    cap.release()
    return motion_score



def compute_face_score(video_path):
    print(f"Trying to open video: {video_path}")
    # Load OpenCV's pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        raise ValueError(f"Error opening video file {video_path}")

    # Variables for computing the score
    total_faces = 0
    total_frames = 0
    total_area = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        total_frames += 1

        # Convert frame to grayscale (Haar Cascade works on grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Count the faces and compute the area
        num_faces = len(faces)
        total_faces += num_faces
        for (x, y, w, h) in faces:
            total_area += w * h  # Area of detected faces

    cap.release()

    # Compute a face score
    if total_frames > 0:
       
        face_score = (total_faces + (total_area / total_frames)) / total_frames
    else:
        face_score = 0  # No frames processed, no faces detected

    return face_score


def create_summary_video(scenes_folder, output_folder, top_k=15):
    scene_files = sorted(os.listdir(scenes_folder))
    
    scene_scores = []

    for scene_file in scene_files:
        if not scene_file.endswith(".mp4"):
            continue
        scene_path = os.path.join(scenes_folder, scene_file)
        score = score_scene(scene_path)
        scene_scores.append((scene_path, score))

    top_scenes = sorted(scene_scores, key=lambda x: x[1], reverse=True)[:top_k]
    def extract_index(filename):
        return int(filename.split("_")[1].split(".")[0])  

    top_scenes_sorted_by_order = sorted(top_scenes, key=lambda x: extract_index(x[0]))

   
    clips = [VideoFileClip(scene[0]) for scene in top_scenes_sorted_by_order]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_folder, codec="libx264")







