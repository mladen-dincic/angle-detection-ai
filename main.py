from ultralytics import YOLO
import cv2
import math

def distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def cosine_rule(point1, point2, point3):
    a = distance(point1, point2)
    b = distance(point2, point3)
    c = distance(point3, point1)

    if a == 0 or b == 0 or c == 0:
        return ''
    
    if a + b <= c or b + c <= a or c + a <= b:
        return ''

    cos_theta = (a**2 + b**2 - c**2) / (2 * a * b)

    theta_rad = math.acos(cos_theta)
    theta_deg = math.degrees(theta_rad)
    
    return round(theta_deg, 2)

model = YOLO('yolov8n-pose.pt')

input = 'videos/man_running.mp4'
cap = cv2.VideoCapture(input)

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'X264'), int(fps), (int(w), int(h)))

while True:
    ret, img = cap.read()

    if not ret:
        break

    results = model(img)

    for result in results:
        im_array = results[0].plot(boxes=False)
        img = cv2.cvtColor(im_array,cv2.COLOR_BGR2BGRA)
        for arr in result.keypoints.xy.numpy():
            keypoints = list(map(lambda points: (int(points[0]), int(points[1])), arr))

            if len(keypoints) > 0:
                text1 = str(cosine_rule(keypoints[5], keypoints[7], keypoints[9])) + 'deg'
                text2 = str(cosine_rule(keypoints[6], keypoints[8], keypoints[10])) + 'deg'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                text_size1 = cv2.getTextSize(text1, font, font_scale, font_thickness)[0]
                text_size2 = cv2.getTextSize(text2, font, font_scale, font_thickness)[0]
                margin_x = 5
                margin_y = 5
                text_box1 = ((keypoints[7][0] - margin_x, keypoints[7][1] - text_size1[1] - margin_y), (keypoints[7][0] + text_size1[0] + margin_x, keypoints[7][1] + margin_y)) #Left hand
                text_box2 = ((keypoints[8][0] - margin_x, keypoints[8][1] - text_size2[1] - margin_y), (keypoints[8][0] + text_size2[0] + margin_x, keypoints[8][1] + margin_y)) #Right hand
                img = cv2.rectangle(img, text_box1[0], text_box1[1], (225, 126, 34), -1) #Left hand
                img = cv2.rectangle(img, text_box2[0], text_box2[1], (225, 126, 34), -1) #Right hand
                img = cv2.putText(img, text1, keypoints[7], font, font_scale, (255, 255, 255), font_thickness) #Left hand
                img = cv2.putText(img, text2, keypoints[8], font, font_scale, (255, 255, 255), font_thickness) #Right hand

    cv2.imshow('Webcam', img)
    out.write(img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()