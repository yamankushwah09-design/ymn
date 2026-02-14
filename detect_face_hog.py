################################################################################
#                        NVIDIA Cloud Lab by AiProff.ai
#  Task 3: HOG Face Detection with System Monitoring
#  Compare HOG vs Haar Cascade (Task 1) to understand differences
################################################################################

# Step 1: Import required libraries - cv2 for computer vision, dlib for HOG, psutil for monitoring
import cv2
import psutil
import subprocess
import time
import re
import sys
import os

# Step 2: Define function to get GPU usage using tegrastats
def get_gpu_usage():
    """
    Get GPU usage statistics from tegrastats command.
    Returns formatted statistics from Jetson Nano's tegrastats utility.
    """
    try:
        process = subprocess.Popen(
            ['tegrastats', '--interval', '1000'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        time.sleep(2)
        process.terminate()
        output, _ = process.communicate(timeout=1)
        
        lines = output.strip().split('\n')
        if lines:
            return lines[-1]
        return "No GPU stats available"
        
    except Exception as e:
        return f"Error reading GPU stats: {e}"

# Step 3: Define function to parse and format GPU stats
def format_gpu_stats(raw_stats):
    """
    Parse tegrastats output and format it into a readable table.
    Extracts RAM, SWAP, CPU cores, frequencies, and temperatures.
    """
    
    if "Error" in raw_stats or not raw_stats:
        return raw_stats
    
    ram_usage = "N/A"
    swap_usage = "N/A"
    cpu_cores = []
    emc_freq = "N/A"
    gpu_freq = "N/A"
    temperatures = {}
    
    try:
        ram_match = re.search(r'RAM (\d+/\d+MB)', raw_stats)
        if ram_match:
            ram_usage = ram_match.group(1)
        
        swap_match = re.search(r'SWAP (\d+/\d+MB) \(cached (\d+MB)\)', raw_stats)
        if swap_match:
            swap_usage = f"{swap_match.group(1)} (cached {swap_match.group(2)})"
        
        cpu_match = re.search(r'CPU \[(.*?)\]', raw_stats)
        if cpu_match:
            cpu_data = cpu_match.group(1).split(',')
            cpu_cores = [core.strip() + "MHz" for core in cpu_data]
        
        emc_match = re.search(r'EMC_FREQ (\d+%)', raw_stats)
        if emc_match:
            emc_freq = emc_match.group(1)
        
        gpu_match = re.search(r'GR3D_FREQ (\d+%)', raw_stats)
        if gpu_match:
            gpu_freq = gpu_match.group(1)
        
        temp_patterns = {
            'PLL': r'PLL@([\d.]+C)',
            'CPU': r'CPU@([\d.]+C)',
            'PMIC': r'PMIC@([\d.]+C)',
            'GPU': r'GPU@([\d.]+C)',
            'AO': r'AO@([\d.]+C)',
            'Thermal': r'thermal@([\d.]+C)'
        }
        
        for temp_name, pattern in temp_patterns.items():
            temp_match = re.search(pattern, raw_stats)
            if temp_match:
                temperatures[temp_name] = temp_match.group(1)
        
    except Exception as e:
        return f"Error parsing GPU stats: {e}"
    
    formatted_output = "\n" + "=" * 72 + "\n"
    formatted_output += " " * 28 + "GPU STATS\n"
    formatted_output += "=" * 72 + "\n"
    
    formatted_output += f"RAM Usage      : {ram_usage}\n"
    formatted_output += f"SWAP Usage     : {swap_usage}\n"
    
    if cpu_cores:
        formatted_output += f"CPU Cores      : [{', '.join(cpu_cores)}]\n"
    
    formatted_output += f"EMC Frequency  : {emc_freq}\n"
    formatted_output += f"GPU Frequency  : {gpu_freq}\n"
    
    if temperatures:
        formatted_output += "Temperatures   :\n"
        for temp_name, temp_value in temperatures.items():
            formatted_output += f"  - {temp_name:<10} : {temp_value}\n"
    
    formatted_output += "=" * 72
    
    return formatted_output

# Print header
print("=" * 72)
print(" " * 18 + "HOG FACE DETECTION WITH MONITORING")
print("=" * 72)
print("This task uses HOG (Histogram of Oriented Gradients) for face detection")
print("Compare with Task 1 (Haar Cascade) to understand the differences!")
print("=" * 72)

# Step 4: Load image
image_path = "image.jpg"
print(f"\n[1/9] Loading image: {image_path}")

if not os.path.exists(image_path):
    print(f"❌ ERROR: Image file '{image_path}' not found!")
    print(f"   Current directory: {os.getcwd()}")
    sys.exit(1)

img = cv2.imread(image_path)
if img is None:
    raise SystemExit("Failed to load 'image.jpg'")

print(f"✅ Image loaded successfully")
print(f"   Dimensions: {img.shape[1]} x {img.shape[0]} pixels")

# Step 5: Convert to grayscale (HOG can work with color, but grayscale is faster)
print("\n[2/9] Converting to grayscale...")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"✅ Converted to grayscale")

# Step 6: Resize image for optimal processing
print("\n[3/9] Resizing image for optimal processing...")
original_shape = img.shape
scale = 800.0 / max(img.shape[:2])
if scale < 1.0:
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    gray = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)))
    print(f"✅ Resized from {original_shape[1]}x{original_shape[0]} to {img.shape[1]}x{img.shape[0]}")
else:
    print(f"✅ No resizing needed (image already optimal size)")

# Step 7: Check if dlib is available, if not, use OpenCV's HOG
print("\n[4/9] Initializing HOG face detector...")

try:
    import dlib
    print("   Attempting to use dlib's HOG face detector...")
    
    # Initialize dlib's HOG-based face detector
    hog_face_detector = dlib.get_frontal_face_detector()
    use_dlib = True
    print("✅ Using dlib's HOG face detector")
    print("   Method: HOG (Histogram of Oriented Gradients) + Linear SVM")
    
except ImportError:
    print("   ⚠️  dlib not available, using alternative method...")
    print("   Installing dlib for better HOG face detection...")
    print("\n   To install dlib, run:")
    print("   pip3 install dlib --break-system-packages")
    print("\n   For now, using OpenCV's DNN face detector as alternative...")
    
    # Use OpenCV's DNN module with pre-trained model as fallback
    use_dlib = False
    
    # Try to use OpenCV's DNN face detector
    try:
        # Download model files if not present
        prototxt_path = "deploy.prototxt"
        model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        
        if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
            print("   Downloading DNN face detection model...")
            import urllib.request
            
            if not os.path.exists(prototxt_path):
                prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
                urllib.request.urlretrieve(prototxt_url, prototxt_path)
                print(f"   ✅ Downloaded {prototxt_path}")
            
            if not os.path.exists(model_path):
                model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
                urllib.request.urlretrieve(model_url, model_path)
                print(f"   ✅ Downloaded {model_path}")
        
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        print("✅ Using OpenCV DNN face detector (alternative to HOG)")
        print("   Method: Deep Neural Network (more accurate than HOG)")
        use_dnn = True
        
    except Exception as e:
        print(f"   ❌ Could not initialize face detector: {e}")
        print("\n   Please install dlib:")
        print("   sudo apt-get install -y python3-dev")
        print("   pip3 install dlib --break-system-packages")
        sys.exit(1)

# Step 8: Detect faces using the selected method
print(f"\n[5/9] Detecting faces...")
start_time = time.time()

if use_dlib:
    # Use dlib's HOG face detector
    # dlib.get_frontal_face_detector() returns a detector object
    # It detects faces and returns rectangles
    faces_dlib = hog_face_detector(gray, 1)  # 1 = upsample image 1 time
    
    # Convert dlib rectangles to OpenCV format (x, y, w, h)
    faces = []
    for face in faces_dlib:
        x = face.left()
        y = face.top()
        w = face.right() - face.left()
        h = face.bottom() - face.top()
        faces.append((x, y, w, h))
    
    detection_method = "HOG (dlib)"
    
else:
    # Use OpenCV DNN face detector
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                  (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            
            faces.append((startX, startY, endX - startX, endY - startY))
    
    detection_method = "DNN (Deep Neural Network)"

detection_time = time.time() - start_time

print(f"✅ Detection complete in {detection_time:.3f} seconds")
print(f"   Method used: {detection_method}")
print(f"   Detected {len(faces)} face(s)")

if len(faces) > 0:
    for i, (x, y, w, h) in enumerate(faces, 1):
        print(f"   Face {i}: Position=({x}, {y}), Size={w}x{h}")

# Step 9: Draw bounding boxes around detected faces
print(f"\n[6/9] Drawing bounding boxes...")

# Use different color for HOG/DNN to distinguish from Haar Cascade
# HOG/DNN = Blue, Haar Cascade (Task 1) = Green
box_color = (255, 0, 0)  # Blue (BGR format)
box_thickness = 2

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), box_color, box_thickness)

if len(faces) > 0:
    print(f"✅ Drew {len(faces)} bounding box(es) in BLUE")
    print(f"   Note: Blue boxes = HOG/DNN detection")
    print(f"         Green boxes (Task 1) = Haar Cascade detection")
else:
    print("⚠️  No faces detected to draw")

# Step 10: Add text label showing detection method
print(f"\n[7/9] Adding detection method label...")
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, f"Method: {detection_method}", (10, 30), 
            font, 0.7, (255, 0, 0), 2)
cv2.putText(img, f"Faces: {len(faces)}", (10, 60), 
            font, 0.7, (255, 0, 0), 2)
cv2.putText(img, f"Time: {detection_time:.3f}s", (10, 90), 
            font, 0.7, (255, 0, 0), 2)
print(f"✅ Labels added to image")

# Step 11: Save output image
output_path = "task3_face_detected_hog.jpg"
print(f"\n[8/9] Saving result to: {output_path}")
cv2.imwrite(output_path, img)
print(f"✅ Output saved successfully")

# Print detection summary with comparison
print("\n" + "=" * 72)
print(" " * 25 + "DETECTION SUMMARY")
print("=" * 72)
print(f"Detection Method       : {detection_method}")
print(f"Faces Detected         : {len(faces)}")
print(f"Detection Time         : {detection_time:.3f} seconds")
print(f"Output File            : {output_path}")
print("=" * 72)

print("\n" + "=" * 72)
print(" " * 20 + "HOG vs HAAR CASCADE COMPARISON")
print("=" * 72)
print("Task 1 (Haar Cascade)          vs          Task 3 (HOG/DNN)")
print("-" * 72)
print("✓ Faster detection                         ✓ More accurate detection")
print("✓ Lower CPU usage                          ✓ Better with varied poses")
print("✓ Works well with frontal faces            ✓ Handles lighting better")
print("✓ Simple & lightweight                     ✓ More robust")
print("✓ Good for real-time apps                  ✓ Better for quality results")
print("-" * 72)
print("Use Haar Cascade when:                     Use HOG/DNN when:")
print("  • Speed is critical                        • Accuracy is critical")
print("  • Limited resources                        • Resources available")
print("  • Simple frontal faces                     • Varied angles/lighting")
print("  • Real-time processing                     • Offline processing")
print("=" * 72)

# Step 12: Collect system resource statistics
print(f"\n[9/9] Collecting system resource statistics...")
print("   (This will take a few seconds...)")

cpu_usage = psutil.cpu_percent(interval=1)
raw_gpu_stats = get_gpu_usage()
formatted_gpu_stats = format_gpu_stats(raw_gpu_stats)

print(f"\nCPU Usage: {cpu_usage}%")
print(formatted_gpu_stats)

# Print final summary
print("\n" + "=" * 72)
print(" " * 27 + "COMPLETE!")
print("=" * 72)
print(f"✅ Detected {len(faces)} face(s) using {detection_method}")
print(f"✅ Detection time: {detection_time:.3f} seconds")
print(f"✅ Output saved to: {output_path}")
print(f"✅ System monitoring complete")
print("=" * 72)
print("\nNEXT STEPS:")
print("  1. Compare output images:")
print(f"     - Task 1: face_detected.jpg (GREEN boxes = Haar Cascade)")
print(f"     - Task 3: {output_path} (BLUE boxes = HOG/DNN)")
print("  2. Compare detection times and accuracy")
print("  3. Notice differences in bounding box precision")
print("=" * 72)

cv2.destroyAllWindows()