import cv2
import easyocr
from deep_translator import GoogleTranslator
from ultralytics import YOLO
import numpy as np
import time
import pyttsx3
import re
from collections import defaultdict
import difflib


engine = pyttsx3.init()
engine.setProperty('rate', 150)
tts_enabled = False

reader = easyocr.Reader(['en'], gpu=False)
model = YOLO('yolov8n.pt')

languages = ['hi', 'en']
lang_names = {'hi': 'Hindi', 'en': 'English'}
lang_index = 0
target_lang = languages[lang_index]


detected_texts = []
last_translation_time = 0
translation_cooldown = 2.0 
last_spoken_text = ""

def preprocess_text(text):
    """Clean and validate OCR text"""
  
    text = re.sub(r'[^\w\s\.\,\!\?\-\:]', '', text)
    
    words = text.split()
    words = [word for word in words if len(word) > 1 or word.lower() in ['i', 'a']]
    
  
    cleaned_text = ' '.join(words).strip()
    

    if len(cleaned_text) < 2:
        return None

    valid_chars = sum(1 for c in cleaned_text if c.isalnum() or c.isspace())
    if valid_chars / len(cleaned_text) < 0.7:  
        return None
    
    return cleaned_text

def is_similar_to_previous(text, threshold=0.8):
    """Check if text is similar to recently detected texts"""
    global detected_texts
    
    for prev_text in detected_texts[-5:]:
        similarity = difflib.SequenceMatcher(None, text.lower(), prev_text.lower()).ratio()
        if similarity > threshold:
            return True
    return False

def combine_nearby_texts(texts_with_boxes, distance_threshold=50):
    """Combine texts that are close to each other to form sentences"""
    if not texts_with_boxes:
        return []
    

    texts_with_boxes.sort(key=lambda x: (x[1][1], x[1][0]))  
    
    combined_groups = []
    current_group = [texts_with_boxes[0]]
    
    for i in range(1, len(texts_with_boxes)):
        current_text, current_box = texts_with_boxes[i]
        last_text, last_box = current_group[-1]
        
      
        y_distance = abs(current_box[1] - last_box[1])  
        x_distance = abs(current_box[0] - last_box[2]) 
    
        if y_distance < distance_threshold and x_distance < distance_threshold * 3:
            current_group.append(texts_with_boxes[i])
        else:

            if current_group:
                combined_groups.append(current_group)
            current_group = [texts_with_boxes[i]]
    
    if current_group:
        combined_groups.append(current_group)

    combined_sentences = []
    for group in combined_groups:
        combined_text = ' '.join([text for text, box in group])
        combined_text = re.sub(r'\s+', ' ', combined_text).strip() 
        
        if len(combined_text.split()) >= 2: 
           
            combined_sentences.append((combined_text, group[0][1]))
    
    return combined_sentences

def safe_translate(text, target_lang='hi'):
    """Enhanced translation with error handling"""
    try:
        if target_lang == 'en': 
            return text
        
        translator = GoogleTranslator(source='en', target=target_lang)
        translated = translator.translate(text)
 
        if translated and translated.strip() and translated != text:
            return translated
        else:
            return f"({text})"  
            
    except Exception as e:
        print(f"⚠️ Translation error for '{text}': {e}")
        return f"({text})"

def speak(text):
    """Enhanced TTS function"""
    try:
   
        speech_text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
        if speech_text.strip():
            engine.say(speech_text)
            engine.runAndWait()
    except Exception as e:
        print(" TTS error:", e)

def enhance_frame_for_ocr(frame):
    """Preprocessing to improve OCR accuracy"""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(blurred)
    
    return enhanced

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open webcam")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(" Enhanced Wearable Translation Glasses running...")
print(" Press 'l' to change language |  Press 't' to toggle speech |  Press 'q' to quit")
print(" Press 's' to capture and process current frame")

prev_time = 0
fps_limit = 1 / 5 
frame_skip_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed.")
        continue

    frame_skip_counter += 1
    if frame_skip_counter % 3 != 0:
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
        continue

    curr_time = time.time()
    if curr_time - prev_time < fps_limit:
        continue
    prev_time = curr_time

    try:

        results = model(frame, verbose=False)
        boxes = results[0].boxes
        
        texts_with_boxes = []

        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                if conf < 0.4: 
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
        
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)
                
                roi = frame[y1:y2, x1:x2]

                if roi.size == 0 or roi.shape[0] < 15 or roi.shape[1] < 15:
                    continue

                enhanced_roi = enhance_frame_for_ocr(roi)
  
                ocr_results = reader.readtext(enhanced_roi)

                for (_, text, ocr_conf) in ocr_results:
                    if ocr_conf > 0.6:  
                        cleaned_text = preprocess_text(text)
                        if cleaned_text and not is_similar_to_previous(cleaned_text):
                            texts_with_boxes.append((cleaned_text, (x1, y1, x2, y2)))

        combined_sentences = combine_nearby_texts(texts_with_boxes)

        for sentence, box in combined_sentences:

            detected_texts.append(sentence)
            if len(detected_texts) > 10: 
                detected_texts.pop(0)

            if target_lang != 'en':
                translated = safe_translate(sentence, target_lang)
            else:
                translated = sentence
            
            print(f" Sentence: '{sentence}'")
            print(f" Translation: '{translated}' ({lang_names[target_lang]})")
            print("-" * 50)

            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
            lines = [sentence[:30], translated[:30]]  
            for i, line in enumerate(lines):
                y_offset = y1 - 30 + (i * 20)
                cv2.putText(frame, line, (x1, max(20, y_offset)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (255, 255, 255) if i == 0 else (0, 255, 0), 2)


            if (tts_enabled and translated != last_spoken_text and 
                curr_time - last_translation_time > translation_cooldown):
                last_spoken_text = translated
                last_translation_time = curr_time
                speak(translated)

    except Exception as e:
        print(" Error during processing:", e)


    status_text = f"Lang: {lang_names[target_lang]} | TTS: {'On' if tts_enabled else 'Off'}"
    cv2.putText(frame, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    

    cv2.putText(frame, "Press 'l' for lang | 't' for TTS | 'q' to quit", 
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("🕶️ Enhanced Translation Glasses", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        lang_index = (lang_index + 1) % len(languages)
        target_lang = languages[lang_index]
        print(f" Language changed to: {lang_names[target_lang]}")
        detected_texts.clear()  
        last_spoken_text = ""
    elif key == ord('t'):
        tts_enabled = not tts_enabled
        print(f" Text-to-Speech {'Enabled' if tts_enabled else 'Disabled'}")
        last_spoken_text = ""

cap.release()
cv2.destroyAllWindows()
print(" Translation Glasses stopped.")