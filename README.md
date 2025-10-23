OBJECT DETECTION APP

This project is an object detection and tracking application built with OpenCV and a Tkinter-based GUI.
The user can choose a video file and a predefined color (or "Any" for all colors), and the program will detect, track, and draw motion trails of the moving objects within the video.

--

🔍 Features

- Video file selection through a graphical Tkinter interface.
- Color selection for detection (Red, Green, Blue, Yellow, Orange, Purple, Any).
- HSV color filtering to isolate the chosen color.
- Background Subtraction (MOG2) for detecting moving objects.
- Morphological operations (open/close) for noise reduction.
- Contour detection and bounding box drawing for identified objects.
- Shape recognition (triangle, rectangle, circle, etc.) based on contour vertices.
- Trail tracking — drawing motion paths of detected objects.
- Real-time display of both the processed frame and the mask side by side.

--

⚙️ Technologies

- Python
- OpenCV (cv2)
- NumPy
- Tkinter (GUI)

--

⚙️ Main Functions & Key Commands
- 🖼️ 1. Video Selection
```file_path = filedialog.askopenfilename()```


Displays a file dialog for the user to select a video.
This is called inside ```select_video()```, which is triggered by:

```button = tk.Button(root, text="Select Video", command=select_video)```

- 🎨 2. Color Selection
```color_combobox = ttk.Combobox(root, values=list(color_ranges.keys()))```


Creates a dropdown menu to choose a color.
The selection is stored in ```selected_color``` via the``` on_color_select(event)``` callback.

- 🧠 3. Object Detection

The ```process_video(cap)``` function performs:

Frame capture: ```ret, frame = cap.read()```

HSV conversion: ```cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)```

Color filtering: ```cv2.inRange()```


Background subtraction:
```backgroundObject = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)```


Contour detection:

```contours, _ = cv2.findContours(...)```

- 🟩 4. Drawing Bounding Boxes and Labels
```cv2.rectangle(frameCopy, (x, y), (x + width, y + height), (0, 255, 0), 2)```
```cv2.putText(frameCopy, shape, (x, y - 10), ...)```


Draws green rectangles and shape labels above detected objects.

- 🔺 5. Shape Detection

The ```detect_shape(contour)``` function uses:

```approx = cv2.approxPolyDP(contour, 0.04 * peri, True)```


to determine object shape based on vertex count (3=triangle, 4=rectangle, >6=circle).

- 🧵 6. Motion Tracking (Trails)

Functions ```update_trails()``` and ```draw_trails()```:

Store detected positions across frames.

Draw motion paths using:

```cv2.line(frame, pt1, pt2, color, 2)```

- 🖥️ 7. Display

The program shows the processed frame and mask side by side:

```cv2.imshow("Object Detection", concatenated_images)```


Press ‘q’ to exit.


📄 License

The project is available for free use and modification for educational purposes.

Thank you for your interest in this project! 🚀****
