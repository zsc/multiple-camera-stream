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
 
def detect_face(face_detector, img, draw=False, skip_ir=True, ratio=3, use_dlib=False):
    if skip_ir:
        gray = cv2.resize(cv2.cvtColor(img[:,:-640], cv2.COLOR_BGR2GRAY), None, fx=1/ratio, fy=1/ratio)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if use_dlib:
        dets = face_detector(gray, 0)
        faces = [(d.left(), d.top(), d.right() - d.left(), d.bottom() - d.top()) for d in dets]
    else:
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    if False and len(faces) > 0:
        cv2.imwrite('/tmp/t.png', img)
        with open('/tmp/t.txt', 'w') as f:
            for (x, y, w, h) in faces:
                f.write('{},{},{},{}\n'.format(x, y, w, h))
    if draw:
        for (x, y, w, h) in faces:
            x *= ratio
            y *= ratio
            w *= ratio
            h *= ratio
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            offset = img.shape[1] - 640
            label = '{:.1f}'.format(img[y:y+h, x+offset:x+offset+w].mean(axis=2).flatten().max() / 255. * 40)
            cv2.rectangle(img, (x + offset, y), (x+offset+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label, (x + offset, y + h), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

def gen_frames():
    use_dlib = True
    if use_dlib:
        import dlib
        face_detector = dlib.get_frontal_face_detector()
    else:
        cascPath = "haarcascade_frontalface_default.xml"
        face_detector = cv2.CascadeClassifier(cascPath) 
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

                #cv2.imwrite('/tmp/frame.jpg', frame)
                #cv2.imwrite('/tmp/frame2.jpg', frame2)
                frame = frame[:,90:1180]
                frame2 = frame2[20:470]
                img = np.concatenate([cv2.resize(frame, (int(frame2.shape[0] / frame.shape[0] * frame.shape[1]), frame2.shape[0])), frame2], axis=1)
                if True:
                    detect_face(face_detector, img, draw=True, use_dlib=use_dlib, ratio=2 if use_dlib else 3)
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
