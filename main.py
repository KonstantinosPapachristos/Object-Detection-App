import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk

# Predefined color ranges in HSV
color_ranges = {
    "Red": ([0, 120, 70], [10, 255, 255]),
    "Green": ([36, 100, 100], [86, 255, 255]),
    "Blue": ([94, 80, 2], [126, 255, 255]),
    "Yellow": ([15, 150, 150], [35, 255, 255]),
    "Orange": ([10, 100, 20], [25, 255, 255]),
    "Purple": ([129, 50, 70], [158, 255, 255]),
    "Any": (None, None)  # Special case for any color
}

selected_color = None
object_trails = []  # List to store the trails of objects
trail_colors = []  # List to store the colors of trails

def select_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        cap = cv2.VideoCapture(file_path)
        process_video(cap)

def process_video(cap):
    global selected_color
    lower_color, upper_color = color_ranges[selected_color]
    kernel = np.ones((5, 5), np.uint8)  # Slightly larger kernel for better noise removal

    # Create a background object with a longer history for stability
    backgroundObject = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the background subtractor to get the foreground mask
        fgmask = backgroundObject.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)

        if selected_color != "Any":
            # Convert the frame to HSV color space
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Create a mask with the specified color range
            color_mask = cv2.inRange(hsv_frame, np.array(lower_color), np.array(upper_color))
            # Combine the color mask and foreground mask
            combined_mask = cv2.bitwise_and(fgmask, color_mask)
        else:
            combined_mask = fgmask

        # Apply morphological operations to reduce noise
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=3)

        # Detect contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frameCopy = frame.copy()
        new_trails = []  # Store the new positions of the detected objects
        new_trail_colors = []  # Store the colors of the new trails

        # Loop inside the contour and search for bigger ones
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:  # Adjust the area threshold as needed
                # Get the area coordinates
                x, y, width, height = cv2.boundingRect(cnt)
                # Draw a rectangle around the area
                cv2.rectangle(frameCopy, (x, y), (x + width, y + height), (0, 255, 0), 2)
                # Detect shape
                shape = detect_shape(cnt)
                # Write the shape name near the object
                cv2.putText(frameCopy, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                # Add the center of the bounding box to the new trails
                new_trails.append((x + width // 2, y + height // 2))
                # Add the color of the trail
                new_trail_colors.append((0, 255, 0))  # Color of the trail (green)

        # Update the trails with the new positions
        update_trails(new_trails, new_trail_colors)

        # Draw the trails
        draw_trails(frameCopy)

        # Resize images
        frameCopy_resized = cv2.resize(frameCopy, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
        combined_mask_resized = cv2.resize(combined_mask, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))

        # Concatenate images horizontally
        concatenated_images = np.hstack((frameCopy_resized, cv2.cvtColor(combined_mask_resized, cv2.COLOR_GRAY2BGR)))

        cv2.imshow("Object Detection", concatenated_images)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def update_trails(new_trails, new_trail_colors):
    global object_trails, trail_colors

    # Remove old trails if there are too many stored
    if len(object_trails) > 20:
        object_trails.pop(0)
        trail_colors.pop(0)

    # If there are no existing trails, add all new trails
    if not object_trails:
        for i in range(len(new_trails)):
            object_trails.append([[new_trails[i]]])
            trail_colors.append([new_trail_colors[i]])
    else:
                # Match new trails with existing trails based on proximity
        matched = [False] * len(new_trails)
        for i in range(len(object_trails[-1])):
            min_dist = float('inf')
            min_idx = -1
            for j in range(len(new_trails)):
                if not matched[j]:
                    dist = np.linalg.norm(np.array(new_trails[j]) - np.array(object_trails[-1][i][-1]))
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = j
            if min_idx != -1 and min_dist < 50:  # Threshold distance for association
                object_trails[-1][i].append(new_trails[min_idx])
                trail_colors[-1][i] = new_trail_colors[min_idx]
                matched[min_idx] = True

        # Add unmatched new trails
        for j in range(len(new_trails)):
            if not matched[j]:
                object_trails[-1].append([new_trails[j]])  # Append a list containing the new trail
                trail_colors[-1].append(new_trail_colors[j])

def draw_trails(frame):
    global object_trails, trail_colors

    for trails, colors in zip(object_trails, trail_colors):
        for trail, color in zip(trails, colors):
            for i in range(1, len(trail)):
                # Convert coordinates to integers
                pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
                pt2 = (int(trail[i][0]), int(trail[i][1]))
                cv2.line(frame, pt1, pt2, color, 2)

def detect_shape(contour):
    # Approximate the contour to get the shape
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

    # Determine the number of vertices
    vertices = len(approx)

    # Return the shape based on the number of vertices
    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        return "Rectangle"
    elif vertices == 5:
        return "Pentagon"
    elif vertices == 6:
        return "Hexagon"
    elif vertices > 6:
        return "Circle"
    else:
        return "Unknown"

def on_color_select(event):
    global selected_color
    selected_color = color_combobox.get()

# Create the GUI
root = tk.Tk()
root.title("Select Video")
root.geometry("300x150")

tk.Label(root, text="Select Color:").pack(pady=10)

color_combobox = ttk.Combobox(root, values=list(color_ranges.keys()))
color_combobox.pack(pady=10)
color_combobox.bind("<<ComboboxSelected>>", on_color_select)

button = tk.Button(root, text="Select Video", command=select_video)
button.pack(pady=20)

root.mainloop()