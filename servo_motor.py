import cv2
import mediapipe as mp
import pickle
import time

# Try serial (ESP8266)
try:
    import serial
    ser = serial.Serial("COM10", 9600)
    time.sleep(2)
    print("ESP8266 Connected")
except:
    ser = None
    print("ESP8266 Not Connected")

# Load model
model = pickle.load(open("model.p", "rb"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Important for Windows

last_sent = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not opening")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)

            prediction = model.predict([landmark_list])
            sign = prediction[0]

            # Send to ESP8266
            if ser is not None and sign != last_sent:
                ser.write(sign.encode())
                last_sent = sign
                print("Sent:", sign)

            cv2.putText(frame, "Sign: " + sign, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Sign Prediction", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()