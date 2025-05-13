import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image
import cv2
  Import python libraries
import numpy as np
import numpy as np
import cv2
from ultralytics import YOLO
  Import python libraries
import numpy as np
from scipy.optimize import linear_sum_assignment
import tempfile



def dprint(*args, **kwargs):
        """Debug print function using inbuilt print
        Args:
            args   : variable number of arguments
            kwargs : variable number of keyword argument
        Return:
            None.
        """
          print(*args, **kwargs)
        pass

class KalmanFilter(object):

        def __init__(self):
            """Initialize variable used by Kalman Filter class
            Args:
                None
            Return:
                None
            """
            self.dt =  0.05  0.005   delta time

            self.A = np.array([[1, 0], [0, 1]])    matrix in observation equations
            self.u = np.zeros((2, 1))    previous state vector

              (x,y) tracking object center
            self.b = np.array([[0], [255]])    vector of observations

            self.P = np.diag((3.0, 3.0))    indicates the confidence in the x and y positions

            self.F = np.array([[1.0, self.dt], [0.0, 1.0]])    state transition mat

            self.Q = np.eye(self.u.shape[0])    process noise matrix
            self.R = np.eye(self.b.shape[0])    observation noise matrix
            self.lastResult = np.array([[0], [255]])

        def predict(self):
            """Predict state vector u and variance of uncertainty P (covariance).
                where,
                u: previous state vector
                P: previous covariance matrix
                F: state transition matrix
                Q: process noise matrix
            Equations:
                u'_{k|k-1} = Fu'_{k-1|k-1}
                P_{k|k-1} = FP_{k-1|k-1} F.T + Q
                where,
                    F.T is F transpose
            Args:
                None
            Return:
                vector of predicted state estimate
            """
              Predicted state estimate
            self.u = np.round(np.dot(self.F, self.u))
              Predicted estimate covariance
            self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
            self.lastResult = self.u    same last predicted result
            return self.u

        def correct(self, b, flag):
            """Correct or update state vector u and variance of uncertainty P (covariance).
            where,
            u: predicted state vector u
            A: matrix in observation equations
            b: vector of observations
            P: predicted covariance matrix
            Q: process noise matrix
            R: observation noise matrix
            Equations:
                C = AP_{k|k-1} A.T + R
                K_{k} = P_{k|k-1} A.T(C.Inv)
                u'_{k|k} = u'_{k|k-1} + K_{k}(b_{k} - Au'_{k|k-1})
                P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
                where,
                    A.T is A transpose
                    C.Inv is C inverse
            Args:
                b: vector of observations
                flag: if "true" prediction result will be updated else detection
            Return:
                predicted state vector u
            """

            if not flag:    update using prediction
                self.b = self.lastResult
            else:    update using detection
                self.b = b
            C = np.dot(self.A, np.dot(self.P, self.A.T)) + self.R
            K = np.dot(self.P, np.dot(self.A.T, np.linalg.inv(C)))

            self.u = np.round(self.u + np.dot(K, (self.b - np.dot(self.A,
                                                                  self.u))))
            self.P = self.P - np.dot(K, np.dot(C, K.T))
            self.lastResult = self.u
            return self.u


class Detectors(object):
        """Detectors class to detect objects in video frame
        Attributes:
            None
        """
        def __init__(self):



             self.model = YOLO("YOLO3.pt")
            self.model = YOLO("last.pt")


              Data transforms (Including Data Augmentation)
            self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

              Load pretrained ResNet18
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

              Replace the final fully connected layer for binary classification
            self.classifier = models.resnet18(pretrained=False)
            num_ftrs = self.classifier.fc.in_features
            self.classifier.fc = nn.Linear(num_ftrs, 2)


            self.classifier.load_state_dict(torch.load("90F1ScoreModel.pth", map_location=torch.device('cpu') ))
            self.classifier.eval()




        def Detect(self, frame):
            """Detect objects in video frame using following pipeline
                - Convert captured frame from BGR to GRAY
                - Perform Background Subtraction
                - Detect edges using Canny Edge Detection
                  http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
                - Retain only edges within the threshold
                - Find contours
                - Find centroids for each valid contours
            Args:
                frame: single video frame
            Return:
                centers: vector of object centroids in a frame
            """


            results = self.model(frame)  self.CLIENT.infer(frame, model_id="anthophora_bomboides-bees/2")
              show contours of tracking objects
              cv2.imshow('Track Bugs', frame)



            results = results[0].boxes.xywh.numpy()  results["predictions"]
            print(results)
            centers = []
            classes = []
            newImg = frame.copy()
            for x , y, w ,h in results:
                x1 = int(x - w / 2)
                y1 = int(y - h/ 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)

                crop = frame[y1:y2,x1:x2,:]

                img_pil = Image.fromarray(crop)

                  Apply your transform
                img_tensor = self.transform(img_pil)

                  Add batch dimension
                img_tensor = img_tensor.unsqueeze(0)

                  Now pass it through your model
                output = self.classifier(img_tensor)

                pollenBearing = np.argmax(output.detach().numpy())

                if pollenBearing:
                    cv2.rectangle(newImg, (x1, y1), (x2, y2), (0,0,0), 2)
                else:
                    cv2.rectangle(newImg, (x1, y1), (x2, y2), (255,255,0), 2)


                b = np.array([[x], [y]])
                centers.append(b)
                classes.append(pollenBearing)


            return centers , newImg , classes
             return centers , frame , classes







class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, prediction, trackIdCount):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.firstDetection = np.asarray(prediction)
        self.classes = []
        self.track_id = trackIdCount    identification of each track object
        self.KF = KalmanFilter()    KF instance to track this object
        self.prediction = np.asarray(prediction)    predicted centroids (x,y)
        self.skipped_frames = 0    number of frames skipped undetected
        self.trace = []    trace path


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """
     (160, 30, 5, 100)
    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount

    def Update(self, detections , classes):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """

          Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

          Calculate cost using sum of square distance between
          predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))     Cost matrix
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                try:
                    diff = self.tracks[i].prediction - detections[j]
                    distance = np.sqrt(diff[0][0]**2 + diff[1][0]**2)
                    cost[i][j] = distance
                except:
                    pass

          Let's average the squared ERROR
        cost = (0.5) * cost
          Using Hungarian Algorithm assign the correct detected measurements
          to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)

         find the optimal assignment of detections to tracks that minimizes the total cost.
        row_ind, col_ind = linear_sum_assignment(cost)  

        for i in range(len(row_ind)):  for each track (existing)
            assignment[row_ind[i]] = col_ind[i]


          Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                  check for cost distance threshold.
                  If cost is very high then un_assign (delete) the track
                if (cost[i][assignment[i]] > self.dist_thresh):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1

          If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)
        if len(del_tracks) > 0:    only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    del self.tracks[id]
                    del assignment[id]
                else:
                    dprint("ERROR: id is greater than length of tracks")

          Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
                if i not in assignment:
                    un_assigned_detects.append(i)

          Start new tracks
        if(len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]],
                              self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

          Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks[i].KF.predict()

            if(assignment[i] != -1):
                 if tracked instance
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct( detections[assignment[i]], 1)
            else:
                 No new tracked instance
                self.tracks[i].prediction = self.tracks[i].KF.correct( np.array([[0], [0]]), 0)

            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) -
                               self.max_trace_length):
                    del self.tracks[i].trace[j]
                    del self.tracks[i].classes[j]

            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].classes.append(classes[assignment[i]])
            self.tracks[i].KF.lastResult = self.tracks[i].prediction







 set to 1 for pipeline images
debug = 0



  Import python libraries
import cv2
import copy
import math
  Variables for drawing
drawing = False    True if mouse is pressed
ix, iy = -1, -1    Initial x, y
fx, fy = -1, -1    Final x, y
rectangle = None

def is_inside(point, box):
    x, y = point
    x_min, y_min, x_max, y_max = box
    return x_min <= x <= x_max and y_min <= y <= y_max

def count_in_out_tracks(box, tracks):

    in_bees = []
    out_bees = []
    in_pollen = []

    for track in tracks:
        trace = track.trace

        if not trace:
            continue    skip empty traces

        start_inside = is_inside(trace[0], box)
        end_inside = is_inside(trace[-1], box)

        if not start_inside and end_inside:
            in_class = np.mean(track.classes)
            if in_class > 0.4:
                in_pollen.append(track.track_id)
            else:
                in_bees.append(track.track_id)

        elif start_inside and not end_inside:
            out_bees.append(track.track_id)

    return in_bees , in_pollen, out_bees


def drawBox(img):
      Load image
    clone = cv2.resize(img.copy(), None, fx=0.4, fy=0.4)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)

    while True:
        temp = clone.copy()

          Draw the rectangle while dragging
        if drawing or rectangle:
            cv2.rectangle(temp, (ix, iy), (fx, fy), (0, 255, 0), 2)

          Display helper text
        cv2.putText(temp, "Press [ENTER] to finalize coords", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('image', temp)

        key = cv2.waitKey(1) & 0xFF

        if key == 13:    Enter key
            if rectangle:
                x1, y1, x2, y2 = rectangle
                x_min, x_max = sorted([x1, x2])
                y_min, y_max = sorted([y1, y2])
                break
    cv2.destroyAllWindows()

    return x_min, y_min, x_max, y_max

  Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        rectangle = (ix, iy, fx, fy)    Save rectangle

def getEntranceCoords(img):
      Load image
     clone = img.copy()
    clone = cv2.resize(img.copy(), None, fx=0.4, fy=0.4)

       Resize by scale factors
     resized_image = cv2.resize(image, None, fx=0.5, fy=0.5)    Resize to 50% of original size


    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)

    while True:
        temp = clone.copy()

          Draw the rectangle while dragging
        if drawing or rectangle:
            cv2.rectangle(temp, (ix, iy), (fx, fy), (0, 255, 0), 2)

          Display helper text
        cv2.putText(temp, "Press [ENTER] to finalize coords", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('image', temp)

        key = cv2.waitKey(1) & 0xFF

        if key == 13:    Enter key
            if rectangle:
                x1, y1, x2, y2 = rectangle
                x_min, x_max = sorted([x1, x2])
                y_min, y_max = sorted([y1, y2])
                break
    cv2.destroyAllWindows()

    return x_min, y_min, x_max, y_max


import cv2
import copy
import math
import pandas as pd


skipFrames = 2
  Variables for drawing
drawing = False    True if mouse is pressed
ix, iy = -1, -1    Initial x, y
fx, fy = -1, -1    Final x, y
rectangle = None
records = []

  (Other functions like is_inside, count_in_out_tracks, draw_rectangle, getEntranceCoords)
 tqdm
import tempfile

def process_video(uploaded_video):
    beeCount = 0
    beeInCount = 0
    beeOutCount = 0

      Create opencv video capture object
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        temp_video_path = tmp.name

      Open the video file with OpenCV
    cap = cv2.VideoCapture(temp_video_path)

    allTracks = set()
    beesIn = set()
    beesOut = set()
    pollenIn = set()

      Create Object Detector
    detector = Detectors()

      Create Object Tracker
    tracker = Tracker(160, 1, 7, 100)

      Variables initialization
    skip_frame_count = 0
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 127, 255),  (127, 0, 255), (127, 0, 127)]
    pause = False

    fps = cap.get(cv2.CAP_PROP_FPS)    Frames per second
    frame_number = 0    To keep track of frames

    ret, prevframe = cap.read()
     prevframe = cv2.resize(prevframe.copy(), None, fx=0.6, fy=0.6)


                                                                                                         
      Ask the user to specify the entrance of the beehive
    box = getEntranceCoords(prevframe)
    box2 =  drawBox(prevframe)
                                                                                                        

      Initialize VideoWriter to save the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')    Codec
    out = cv2.VideoWriter('slow55.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))    (Width, Height)

      Infinite loop to process video frames
    while(True):

        frame_number += 1    Increment frame count
          Calculate time
        seconds = frame_number / fps
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
          Format timestamp nicely
        timestamp = f"{minutes:02d}:{seconds:02d}"

          Capture frame-by-frame
        ret, currentFrame = cap.read()
        currentFrame = cv2.resize(currentFrame.copy(), None, fx=0.4, fy=0.4)
            Draw entrance rectangle
        cv2.rectangle(currentFrame, box2[:2], box2[2:], (255, 255, 255), -1)

        if not ret:
            break

          Make copy of original frame
        orig_frame = copy.copy(currentFrame)

          Skip initial frames that display logo
        if (skip_frame_count < 15):
            skip_frame_count += 1
            continue

          Detect and return centroids of the objects in the frame
        centers, currentFrame, classes = detector.Detect(currentFrame)

          If centroids are detected then track them
        if len(centers) > 0:
              Track object using Kalman Filter
            tracker.Update(centers, classes)
            beesTrackedIn, pollenTrackedIn, beesTrackedOut = count_in_out_tracks(box, tracker.tracks)

              Update bees and pollen counts
            if len(beesTrackedIn) > 0:
                for bee in beesTrackedIn:
                    beesIn.add(bee)
                    allTracks.add(bee)
            if len(beesTrackedOut) > 0:
                for bee in beesTrackedOut:
                    beesOut.add(bee)
                    allTracks.add(bee)
            if len(pollenTrackedIn) > 0:
                for pollen in pollenTrackedIn:
                    pollenIn.add(pollen)
                    allTracks.add(pollen)

            records.append([timestamp, len(beesIn),len(pollenIn),len(beesOut)])

          Draw tracking lines for each object
        for i in range(len(tracker.tracks)):
            if len(tracker.tracks[i].trace) > 1:
                for j in range(len(tracker.tracks[i].trace) - 1):
                    x1 = tracker.tracks[i].trace[j][0][0]
                    y1 = tracker.tracks[i].trace[j][1][0]
                    x2 = tracker.tracks[i].trace[j+1][0][0]
                    y2 = tracker.tracks[i].trace[j+1][1][0]
                    clr = tracker.tracks[i].track_id % 9
                    cv2.line(currentFrame, (int(x1), int(y1)), (int(x2), int(y2)),
                             track_colors[clr], 2)
                    mid_x = int((x1 + x2) / 2)
                    mid_y = int((y1 + y2) / 2)
                    cv2.putText(currentFrame, str(tracker.tracks[i].classes[j]), (mid_x, mid_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_colors[clr], 2)

          Add dashboard texts
        cv2.putText(currentFrame, f"Total Count: {len(allTracks)}", (10, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(currentFrame, f"Bees In: {len(beesIn)}", (10, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(currentFrame, f"Pollen In: {len(pollenIn)}", (10, 100),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(currentFrame, f"Bees Out: {len(beesOut)}", (10, 130),   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(currentFrame, f"Time: {timestamp}", (10, 160),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

          Draw entrance rectangle
        cv2.rectangle(currentFrame, box[:2], box[2:], (255, 255, 255), 2)
        cv2.imshow('frame',currentFrame)
          Write the frame to the video output
        out.write(currentFrame)

          Slow down the FPS
        cv2.waitKey(50)

      When everything done, release the capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(records, columns=["Time Stamp", "BeesIn", "pollenIn", "BeesOut"])


    return df

import streamlit as st
import pandas as pd
import base64

  --- Streamlit Config ---
st.set_page_config(
    page_title="Beehive AI Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

  --- Custom CSS Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');

    .stApp {
        background-color:  FFF9C4; /* light yellow honey-themed */
    }

    .header-container {
        display: flex;
        justify-content: center; /* Center the logo horizontally */
        padding: 10px 0;
    }

    .header-container img {
        width: 700px; /* Bee Logo size */
        height: auto;
    }

    .bee-text {
        font-family: Pacifico;
        font-size: 30px;
        color:  F57F17;
    }

    .main-title {
        font-size: 38px;
        font-family: Pacifico;
        color:  F57F17;
        text-align: center;
        padding: 10px;
        background-color:  FFFDE7;
        border-radius: 15px;
        margin-top: 10px;
    }

    .section {
        background-color:  FFFFFF;
        padding: 20px;
        border-radius: 15px;
        margin-top: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }

    .footer {
        font-size: 14px;
        text-align: center;
        color: gray;
        margin-top: 50px;
    }

    .bee-image {
        width: 100% !important; /* Allow full width of the column */
        max-width: 1500px; /* Maximum width to prevent excessive scaling */
        height: auto;
    }
    </style>
""", unsafe_allow_html=True)

  --- Load and Display SVG ---
def get_svg_base64(svg_path):
    with open(svg_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def render_svg_image(svg_path, width="300px", height="auto"):
    with open(svg_path, "rb") as f:
        svg_bytes = f.read()
        encoded = base64.b64encode(svg_bytes).decode()
        img_tag = f"<img src='data:image/svg+xml;base64,{encoded}' class='bee-image' />"
        st.markdown(img_tag, unsafe_allow_html=True)

  --- App Title and Logo (Centered) ---
logo_svg = get_svg_base64("bee_logo.svg")
header_html = f"""
<div class="header-container">
    <img src="data:image/svg+xml;base64,{logo_svg}" style="width:300px; height:auto;" alt="Bee Logo" />
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

st.markdown('<div class="main-title"> SQU Beehive Monitor</div>', unsafe_allow_html=True)

  --- Main Interface Section ---
with st.container():

        st.subheader("📤 Upload Beehive Video")


        uploaded_video = st.file_uploader("Upload your beehive video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

        if uploaded_video:
            st.video(uploaded_video)
            st.success("✅ Video uploaded!")

            if st.button("🚀 Start Analysis", use_container_width=True):
                    with st.spinner("Processing video..."):
                        log_data = process_video(uploaded_video)

                    st.success("✅ Analysis complete!")
                    st.markdown("    📊 Bee Activity Log")
                    st.dataframe(log_data)

                      Download button for Excel
                    output_path = "bee_activity_log.xlsx"
                    log_data.to_excel(output_path, index=False)
                    with open(output_path, "rb") as f:
                        st.download_button("⬇️ Download Excel Log", f, file_name="bee_activity_log.xlsx",use_container_width=True)



        st.markdown('</div>', unsafe_allow_html=True)

  --- Footer ---
st.markdown('<div class="footer">Made for Beekeepers </div>', unsafe_allow_html=True)



streamlit run app.py

