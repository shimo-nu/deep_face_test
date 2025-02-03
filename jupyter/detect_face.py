from deepface import DeepFace
from contextlib import contextmanager
import os, sys

def put_text_with_outline(img,text,point,font,size,color,thickness,outline_color):
  img = cv2.putText(img, text, point, font, size, outline_color, thickness + 3)

  img = cv2.putText(img, text, point, font, size, color, thickness)
  return img

@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def analyze_face(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    # with suppress_output():
    try:
        dfs = DeepFace.analyze(
            img_path = img,
            actions = ['age', 'gender', 'race', 'emotion'],
            detector_backend = model,
            enforce_detection = True,
            silent = True
        )
    except ValueError as e:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    # except Exception as e:
        
    #     dfs = DeepFace.analyze(
    #         img_path = img,
    #         actions = ['age', 'gender', 'race', 'emotion'],
    #         detector_backend='ssd',
    #         silent = True
    #     )
    bg_color = np.array([255, 255, 255])
    offset = 0
    for dfs_child in dfs:
        age = dfs_child['age']
        gender = f"Women" if dfs_child['gender']['Woman'] > dfs_child['gender']['Man'] else f"Men"
        # gender = f"Women({int(dfs[0]['gender']['Woman']) / 100})" if dfs[0]['gender']['Woman'] > dfs[0]['gender']['Man'] else f"Men({int(dfs[0]['gender']['Man']) / 100})"
        emotion = dfs_child['dominant_emotion']
        # p1, p2 = [dfs_child['region']['x'] + int(dfs_child['region']['w'] / 2), dfs_child['region']['y'] - int(dfs_child['region']['h'] / 2)], [dfs_child['region']['x'] + int(dfs_child['region']['w'] / 2), dfs_child['region']['y'] + int(dfs_child['region']['h'] / 2)]
        p1, p2 = [dfs_child['region']['x'], dfs_child['region']['y']], [dfs_child['region']['x'] + dfs_child['region']['w'], dfs_child['region']['y'] + dfs_child['region']['h']]
        # p1, p2 = [tmp_p1[1], tmp_p1[0]], [tmp_p2[1], tmp_p2[0]]
        put_text_with_outline(img, f"age     : {age}", (0, 50 + offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2, bg_color.tolist())
        put_text_with_outline(img, f"gender  : {gender}", (0, 80 + offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2, bg_color.tolist())
        put_text_with_outline(img, f"emotion : {emotion}", (0, 110 + offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2, bg_color.tolist())
        
        cv2.rectangle(img, p1, p2, bg_color.tolist(), 2, cv2.LINE_AA)
        
        bg_color -= np.array([100, 50, 50])
        offset += 90
    # plt.imshow(img, cmap='gray')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

import cv2
from rich.progress import Progress
import numpy as np

model = 'retinaface'

input_video_path = "./data/labcam206/20241218_lab206.mp4"
filename = os.path.splitext(os.path.basename(input_video_path))[0]
output_video_path = f"./data/results/{filename}_{model}.mp4"

cap = cv2.VideoCapture(input_video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
fps = int(cap.get(cv2.CAP_PROP_FPS))  
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
frame_size = (width, height)


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

with Progress() as progress:
    task = progress.add_task(
        "[cyan]Processing video... (Frame {}/{}):", total=total_frames
    )
    current_frame = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)  # 再び3チャンネルに変換

        processed_frame = analyze_face(frame, model)
        
        out.write(processed_frame)

        
        progress.update(task, advance=1, description=f"[cyan]Processing video... (Frame {current_frame}/{total_frames})")
        current_frame += 1

cap.release()
out.release()
cv2.destroyAllWindows()
