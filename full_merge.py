import cv2
from skimage.metrics import structural_similarity
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os


def calculate_frame_similarity(prev_frame, current_frame):
    # Converting it to a "grayscale" a.k.a. black and white.
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculating the similarity percentage to the last frame
    similarity = structural_similarity(prev_gray, current_gray)

    return similarity

def get_video_files(directory):
    video_files = []
    extensions = ['.mp4', '.avi', '.mov']  # The supperted file extentions if needed just add more ex. [".webmv"] 
    for file in os.listdir(directory):
        if file.endswith(tuple(extensions)):
            video_files.append(os.path.join(directory, file))

    return video_files

# Function to compute frame embeddings
def compute_frame_embedding(frame):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    encoded_image = model.encode(pil_image)
    return encoded_image
# Load the CLIP model
model = SentenceTransformer('clip-ViT-B-32')


def keep_two_digits(array):
    modified_array = []
    for num in array:
        num_str = str(num)
        first_two_digits = num_str[:2]
        modified_num = float(first_two_digits)
        modified_array.append(modified_num)
    return modified_array



files_list = []
for video_path in files_list:
    cap = cv2.VideoCapture(video_path)


    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video file")
        exit()


    # Compute embeddings for the first frame
    prev_embedding = compute_frame_embedding(prev_frame)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)


    percent1 = []
    percent2 = []
    frame_counter = 0


    while True:
        # Read the next frame
        ret, current_frame = cap.read()
        if not ret:
            break

        # Compute embeddings for the current frame
        current_embedding = compute_frame_embedding(current_frame)

        similarity1 = calculate_frame_similarity(prev_frame, current_frame)
        similarity_percent1 = similarity1 * 100
        percent1.append(similarity_percent1)




        similarity2 = util.pytorch_cos_sim(prev_embedding, current_embedding)[0][0]
        similarity_percent2 = similarity2.item() * 100
        percent2.append(similarity_percent2)


        # Show the current frame disable if not needed
        cv2.imshow("Current Frame", current_frame)



        # Break the loop if 'q' is pressed just in case
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        # Update the previous frame and embedding
        prev_frame = current_frame
        prev_embedding = current_embedding

    percent1 = keep_two_digits(percent1)
    percent2 = keep_two_digits(percent2)
    try:
        filename1 = "array_data1.txt"
        np.savetxt(filename1, percent1)
        filename2 = "array_data2.txt"
        np.savetxt(filename2, percent2)
    except:
        print("scrittura dati fallita")

    cap.release()
    cv2.destroyAllWindows()


    scene = []
    last_trigger = 0
    last_frame_point = 0


    for frame ,simil in enumerate(percent1):




        if(simil < 60):
            if(last_trigger > frame -200):
                pass
            else:
                scene.append([last_frame_point, frame])
                last_trigger = frame # To mine own dearest future self, I beseech thee, pray withhold thy profane remarks, forsooth! Wouldst thou not abstain from thy unseemly comment upon the grievous fact that I, in mine wisdom, hath fashioned twain perfectly identical variables, each fulfilling the selfsame infernal task? Verily, I did so in mine sagacity, foreseeing the likelihood of thy forgetfulness concerning their purpose.
                last_frame_point= frame # to my dearest future self could you not fucking comment the fact that i have made 2 perfectly identical variables doing the same fucking job i did it for you since i know you will most likely forget what their use will be for
    scene.append([last_frame_point, frame])


    consequent = 0
    num = 0
    for start_frame, end_frame in scene:
        for _ in range(start_frame, end_frame):
            if(percent1[_]>90 and percent2[_]>85):
                consequent += 1
        if(consequent > 30):
            scene.pop(num)
            num -= 1
        consequent = 0
        num += 1





    output_folder = 'clips_' + video_path
    output_prefix = '_clip_'
    output_suffix = '.mp4'


    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    video_reader = cv2.VideoCapture(video_path)




    frame_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    current_clip = 0
    for start_frame, end_frame in scene:
        # Create a new video writer for each clip
        output_video = os.path.join(output_folder, f"{video_path}{output_prefix}{current_clip}{output_suffix}")
        video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        # Set the video to the start frame of the clip
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Write frames to the output video.
        for current_frame in range(start_frame, end_frame):
            ret, frame = video_reader.read()
            if not ret:
                break
            video_writer.write(frame)

        video_writer.release()
        current_clip += 1
    print("fille_finished")

    video_reader.release() 