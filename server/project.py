import cv2
import numpy as np

def get_component_label(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h

    capacitor_ratio_range = (0.5, 2.0)
    led_ratio_range = (0.5, 2.0)
    chip_ratio_range = (0.2, 0.8)
    resistor_ratio_range = (0.2, 1.0)
    transistor_ratio_range = (0.5, 2.0)

    if capacitor_ratio_range[0] <= aspect_ratio <= capacitor_ratio_range[1]:
        return "Capacitor"
    elif led_ratio_range[0] <= aspect_ratio <= led_ratio_range[1]:
        return "LED"
    elif chip_ratio_range[0] <= aspect_ratio <= chip_ratio_range[1]:
        return "Chip"
    elif resistor_ratio_range[0] <= aspect_ratio <= resistor_ratio_range[1]:
        return "Resistor"
    elif transistor_ratio_range[0] <= aspect_ratio <= transistor_ratio_range[1]:
        return "Transistor"
    else:
        return "Unknown"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    edges = cv2.Canny(thresh, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            
            component_label = get_component_label(contour)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, component_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Webcam Electrical Component Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
