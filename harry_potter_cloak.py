import cv2
import numpy as np
import time

# ----- Optimized Parameters for Better FPS -----
CAPTURE_BG_SECONDS = 3
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# Balanced HSV color range for RED cloak (good detection + skin protection)
# Moderate ranges that detect red well but avoid most skin tones
lower_red_1 = np.array([0, 120, 80])   # Balanced saturation and value
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([170, 120, 80]) # Balanced saturation and value
upper_red_2 = np.array([180, 255, 255])

# Improved skin tone exclusion ranges
skin_lower = np.array([0, 15, 50])     # More specific skin tone range
skin_upper = np.array([25, 120, 200])  # More specific skin tone range

# If you use GREEN cloak, comment above and uncomment below
# lower_green = np.array([40, 50, 50])
# upper_green = np.array([90, 255, 255])

# ----- Optimized Camera Setup -----
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPEG for better performance

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Test actual FPS
print("Testing camera FPS...")
test_frames = 0
test_start = time.time()
while test_frames < 30:
    ret, frame = cap.read()
    if ret:
        test_frames += 1
test_fps = test_frames / (time.time() - test_start)
print(f"Camera FPS: {test_fps:.1f}")

time.sleep(0.5)
print(f"Capturing background for {CAPTURE_BG_SECONDS} seconds... Please move out of the frame.")

# Optimized background capture
bg_frames = []
start = time.time()
frame_count = 0

while time.time() - start < CAPTURE_BG_SECONDS:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    
    # Reduced blur for better FPS
    frame_blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    bg_frames.append(frame_blurred)
    
    frame_count += 1

# Create median background
if bg_frames:
    background = np.median(bg_frames, axis=0).astype(np.uint8)
    print("Background captured! Now wear your cloak. Press 'q' to quit, 'r' to recapture.")
else:
    print("Error: No background frames captured")
    cap.release()
    exit()

# Pre-allocate arrays for better performance
hsv_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
mask1 = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
mask2 = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
mask_inv = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

# Optimized kernels
kernel_small = np.ones((3, 3), np.uint8)
kernel_medium = np.ones((5, 5), np.uint8)

# Variables for temporal smoothing (reduced for better FPS)
previous_mask = None
mask_history = []

# FPS monitoring variables
fps_counter = 0
fps_start_time = time.time()
current_fps = 0

print("Balanced invisibility cloak ready!")
print("Features: Good red detection, smart skin protection, high FPS")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Optimized preprocessing - reduced blur
    frame_blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    
    # Convert to HSV using pre-allocated array
    cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV, dst=hsv_frame)
    
    # Create masks using pre-allocated arrays
    cv2.inRange(hsv_frame, lower_red_1, upper_red_1, dst=mask1)
    cv2.inRange(hsv_frame, lower_red_2, upper_red_2, dst=mask2)
    cv2.add(mask1, mask2, dst=mask)
    
    # Create skin tone mask to exclude false positives
    skin_mask = cv2.inRange(hsv_frame, skin_lower, skin_upper)
    
    # Apply skin tone filtering more intelligently
    # Only remove skin areas if they are small patches (likely face parts)
    skin_contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in skin_contours:
        area = cv2.contourArea(contour)
        if area < 1000:  # Only remove small skin patches
            cv2.fillPoly(mask, [contour], 0)
    
    # Additional safety: Remove very small areas that might be noise
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200:  # Only remove very small noise areas
            cv2.fillPoly(mask, [contour], 0)
    
    # Optimized morphological operations (reduced iterations)
    cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, dst=mask, iterations=1)
    cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, dst=mask, iterations=1)
    cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_small, dst=mask, iterations=1)
    
    # Simplified temporal smoothing for better FPS
    if previous_mask is not None:
        cv2.addWeighted(mask, 0.8, previous_mask, 0.2, 0, dst=mask)
    
    previous_mask = mask.copy()
    
    # Create inverse mask
    cv2.bitwise_not(mask, dst=mask_inv)
    
    # Optimized blending - use simpler approach for better FPS
    mask_blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    mask_inv_blurred = cv2.GaussianBlur(mask_inv, (3, 3), 0)
    
    # Efficient blending using numpy operations
    mask_normalized = mask_blurred.astype(np.float32) / 255.0
    mask_inv_normalized = mask_inv_blurred.astype(np.float32) / 255.0
    
    # Vectorized blending for better performance
    final = (background * mask_normalized[..., np.newaxis] + 
             frame * mask_inv_normalized[..., np.newaxis]).astype(np.uint8)
    
    # Light smoothing for quality
    final = cv2.bilateralFilter(final, 5, 50, 50)
    
    # Calculate and display FPS
    fps_counter += 1
    if fps_counter % 30 == 0:
        fps_end_time = time.time()
        current_fps = 30 / (fps_end_time - fps_start_time)
        fps_start_time = fps_end_time
    
    # Display FPS on frame
    cv2.putText(final, f"FPS: {current_fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("High-FPS Invisibility Cloak", final)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Recapture background
        print("Recapturing background... Please move out of the frame.")
        bg_frames = []
        start = time.time()
        
        while time.time() - start < CAPTURE_BG_SECONDS:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            frame_blurred = cv2.GaussianBlur(frame, (3, 3), 0)
            bg_frames.append(frame_blurred)
        
        if bg_frames:
            background = np.median(bg_frames, axis=0).astype(np.uint8)
            print("Background recaptured!")
            # Reset temporal smoothing
            previous_mask = None
            mask_history = []

cap.release()
cv2.destroyAllWindows()
print("Application closed.")
