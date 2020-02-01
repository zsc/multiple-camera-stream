from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

def find_camera(id):
    cameras = ['rtsp://192.168.10.15:554/user=admin&password=?channel=1&stream=0.sdp?',
    'rtsp://192.168.10.123']
    return cameras[int(id)]
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
#  for webcam use zero(0)
 

def gen_frames():
     
    cam = find_camera(0)
    cap = cv2.VideoCapture(cam)
    cam2 = find_camera(1)
    cap2 = cv2.VideoCapture(cam2)

    num_frame = 50
    buf = []
    
    while True:
        # for cap in caps:
        # # Capture frame-by-frame
        success, frame = cap.read()  # read the camera frame
        success2, frame2 = cap2.read()  # read the camera frame
        if not success or not success2:
            break
        else:
            buf.append(frame2)
            if len(buf) > num_frame:
                buf = buf[-num_frame:]
                frame2 = buf[0]
                img = np.concatenate([cv2.resize(frame, (int(frame2.shape[0] / frame.shape[0] * frame.shape[1]), frame2.shape[0])), frame2], axis=1)
                ret, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed/<string:id>/', methods=["GET"])
def video_feed(id):
   
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
