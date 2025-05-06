import os
import cv2

def convert_sequence_to_video(input_folder, output_video_path, fps=25):
    images = sorted([f for f in os.listdir(input_folder) if f.endswith(".jpg")])
    if not images:
        print(f"No images in {input_folder}")
        return

    first_frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, _ = first_frame.shape

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        img_path = os.path.join(input_folder, image)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()
    print(f"✅ Created video: {output_video_path}")

# Change this path
base_path = r"D:\AIML\Repos\YOLO\CT3_Mini_Project\data"

for mode in ['training', 'validation']:
    input_base = os.path.join(base_path, mode, "sequences")
    output_base = os.path.join(base_path, mode, "videos")
    os.makedirs(output_base, exist_ok=True)

    for folder in os.listdir(input_base):
        full_input_path = os.path.join(input_base, folder)
        if os.path.isdir(full_input_path):
            output_video = os.path.join(output_base, f"{folder}.mp4")
            convert_sequence_to_video(full_input_path, output_video)
