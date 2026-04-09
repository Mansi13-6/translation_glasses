import cv2
import easyocr
from deep_translator import GoogleTranslator
from ultralytics import YOLO
import numpy as np
import time
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)
tts_enabled = False

reader = easyocr.Reader(['en'], gpu=True)
model = YOLO('yolov8n.pt')

languages = ['hi', 'en'] 
lang_names = {'hi': 'Hindi', 'en': 'English'}
lang_index = 0
target_lang = languages[lang_index]

def safe_translate(text, target_lang='hi'):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        print(f" Translation error for '{text}': {e}")
        return "(translation error)"

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(" TTS error:", e)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Failed to open webcam")
    exit()

print(" Wearable Translation Glasses running...")
print(" Press 'l' to change language |  Press 't' to toggle speech |  Press 'q' to quit")

prev_time = 0
fps_limit = 1 / 10
last_spoken_text = ""  

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Frame capture failed.")
        continue

    curr_time = time.time()
    if curr_time - prev_time < fps_limit:
        continue
    prev_time = curr_time

    try:
        results = model(frame, verbose=False)
        boxes = results[0].boxes

        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]

                if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                    continue

                ocr_results = reader.readtext(roi)

                for (_, text, ocr_conf) in ocr_results:
                    if ocr_conf > 0.5 and text.strip():
                        translated = safe_translate(text, target_lang)
                        print(f" '{text}'  '{translated}' ({lang_names[target_lang]})")

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                        cv2.putText(frame, translated, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        if tts_enabled and translated != last_spoken_text:
                            last_spoken_text = translated
                            speak(translated)

    except Exception as e:
        print(" Error during processing:", e)

    cv2.putText(frame, f"Lang: {lang_names[target_lang]} | TTS: {'On' if tts_enabled else 'Off'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("🕶️ Wearable Translation Glasses", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        lang_index = (lang_index + 1) % len(languages)
        target_lang = languages[lang_index]
        print(f" Language changed to: {lang_names[target_lang]}")
    elif key == ord('t'):
        tts_enabled = not tts_enabled
        print(f" Text-to-Speech {'Enabled' if tts_enabled else 'Disabled'}")
        last_spoken_text = ""  

cap.release()
cv2.destroyAllWindows()

