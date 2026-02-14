################################################################################
#                        NVIDIA Cloud Lab by AiProff.ai
#      Task 2: Face Detection with System Monitoring (Reference Code)
################################################################################

# Step 1: Import required libraries - cv2 for computer vision, psutil for CPU monitoring, subprocess for GPU stats, re for parsing
import cv2
import psutil
import subprocess
import time
import re
import sys
import os

# Step 2: Define function to get GPU usage using tegrastats with timeout approach (Python 3.6 compatible)
def get_gpu_usage():
    """
    Get GPU usage statistics from tegrastats command.
    
    Step-by-step process:
    1. Start tegrastats process with 1 second interval
    2. Let it run for 2 seconds to collect data
    3. Terminate the process
    4. Parse and return the output
    """
    try:
        # Start tegrastats process without --count flag
        # Using universal_newlines=True instead of text=True for Python 3.6 compatibility
        process = subprocess.Popen(
            ['tegrastats', '--interval', '1000'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Let it run for 2 seconds to collect stats
        time.sleep(2)
        
        # Terminate the process
        process.terminate()
        
        # Get all output
        output, _ = process.communicate(timeout=1)
        
        # Return the last line which contains the most recent stats
        lines = output.strip().split('\n')
        if lines:
            return lines[-1]
        return "No GPU stats available"
        
    except Exception as e:
        return f"Error reading GPU stats: {e}"

# Step 3: Define function to parse and format GPU stats into readable table format
def format_gpu_stats(raw_stats):
    """
    Parse tegrastats output and format it into a nice table.
    
    Step-by-step parsing:
    1. Extract RAM usage (e.g., "2295/3956MB")
    2. Extract SWAP usage (e.g., "661/1978MB (cached 47MB)")
    3. Extract CPU core stats (e.g., "[4%@102,2%@102,2%@102,4%@102]")
    4. Extract frequency stats (EMC_FREQ, GR3D_FREQ)
    5. Extract all temperature readings (PLL, CPU, PMIC, GPU, AO, thermal)
    """
    
    if "Error" in raw_stats or not raw_stats:
        return raw_stats
    
    # Initialize variables for parsed data
    ram_usage = "N/A"
    swap_usage = "N/A"
    cpu_cores = []
    emc_freq = "N/A"
    gpu_freq = "N/A"
    temperatures = {}
    
    try:
        # Step 3.1: Extract RAM usage using regex pattern
        ram_match = re.search(r'RAM (\d+/\d+MB)', raw_stats)
        if ram_match:
            ram_usage = ram_match.group(1)
        
        # Step 3.2: Extract SWAP usage with cached info
        swap_match = re.search(r'SWAP (\d+/\d+MB) \(cached (\d+MB)\)', raw_stats)
        if swap_match:
            swap_usage = f"{swap_match.group(1)} (cached {swap_match.group(2)})"
        
        # Step 3.3: Extract CPU core statistics
        cpu_match = re.search(r'CPU \[(.*?)\]', raw_stats)
        if cpu_match:
            cpu_data = cpu_match.group(1).split(',')
            cpu_cores = [core.strip() + "MHz" for core in cpu_data]
        
        # Step 3.4: Extract EMC frequency
        emc_match = re.search(r'EMC_FREQ (\d+%)', raw_stats)
        if emc_match:
            emc_freq = emc_match.group(1)
        
        # Step 3.5: Extract GPU (GR3D) frequency
        gpu_match = re.search(r'GR3D_FREQ (\d+%)', raw_stats)
        if gpu_match:
            gpu_freq = gpu_match.group(1)
        
        # Step 3.6: Extract all temperature readings
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
    
    # Step 3.7: Format everything into a beautiful table
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
print(" " * 15 + "FACE DETECTION WITH SYSTEM MONITORING")
print("=" * 72)

# Step 4: Load image using cv2.imread() which reads image from file path and returns numpy array
image_path = "image.jpg"
print(f"\n[1/8] Loading image: {image_path}")

if not os.path.exists(image_path):
    print(f"❌ ERROR: Image file '{image_path}' not found!")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Files in directory: {os.listdir('.')}")
    sys.exit(1)

img = cv2.imread(image_path)
if img is None:
    raise SystemExit("Failed to load 'image.jpg' — make sure file exists in the script folder.")

print(f"✅ Image loaded successfully")
print(f"   Dimensions: {img.shape[1]} x {img.shape[0]} pixels")

# Step 5: Convert image to grayscale using cv2.cvtColor() as Haar Cascade works better on grayscale images
print("\n[2/8] Converting to grayscale...")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"✅ Converted to grayscale")

# Step 6: Resize image for faster processing (optional but improves performance)
print("\n[3/8] Resizing image for optimal processing...")
original_shape = img.shape
scale = 800.0 / max(img.shape[:2])
if scale < 1.0:
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    gray = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)))
    print(f"✅ Resized from {original_shape[1]}x{original_shape[0]} to {img.shape[1]}x{img.shape[0]}")
else:
    print(f"✅ No resizing needed (image already optimal size)")

# Step 7: Load pre-trained Haar Cascade classifier for face detection using cv2.CascadeClassifier() with XML file path
cascade_path = "haarcascade_frontalface_default.xml"
print(f"\n[4/8] Loading Haar Cascade: {cascade_path}")

if not os.path.exists(cascade_path):
    print(f"❌ ERROR: Cascade file '{cascade_path}' not found!")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Files in directory: {os.listdir('.')}")
    print("\n   Make sure the cascade file is in the same directory as this script.")
    sys.exit(1)

face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise SystemExit(f"Failed to load cascade at {cascade_path}")

print(f"✅ Cascade loaded successfully")

# Step 8: Detect faces using detectMultiScale() with scaleFactor for pyramid scaling, minNeighbors for detection quality
print("\n[5/8] Detecting faces...")
start_time = time.time()
faces = face_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.1, 
    minNeighbors=5, 
    minSize=(30, 30)
)
detection_time = time.time() - start_time

print(f"✅ Detection complete in {detection_time:.3f} seconds")
print(f"   Detected {len(faces)} face(s)")

if len(faces) > 0:
    for i, (x, y, w, h) in enumerate(faces, 1):
        print(f"   Face {i}: Position=({x}, {y}), Size={w}x{h}")

# Step 9: Draw green bounding boxes around detected faces using cv2.rectangle() with BGR color format (0,255,0)
print("\n[6/8] Drawing bounding boxes...")
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

if len(faces) > 0:
    print(f"✅ Drew {len(faces)} bounding box(es)")
else:
    print("⚠️  No faces detected to draw")

# Step 10: Save output image to disk using cv2.imwrite()
output_path = "task2_face_detected.jpg"
print(f"\n[7/8] Saving result to: {output_path}")
cv2.imwrite(output_path, img)
print(f"✅ Output saved successfully")

print(f"\nDetected {len(faces)} face(s). Output saved to {output_path}")

# Step 11: Get CPU usage percentage using psutil.cpu_percent() and GPU stats using custom get_gpu_usage() function
print("\n[8/8] Collecting system resource statistics...")
print("   (This will take a few seconds...)")

# Get CPU usage with 1 second interval for accurate measurement
cpu_usage = psutil.cpu_percent(interval=1)

# Get GPU stats from tegrastats
raw_gpu_stats = get_gpu_usage()

# Step 12: Format the raw GPU stats into readable table format
formatted_gpu_stats = format_gpu_stats(raw_gpu_stats)

# Step 13: Print system resource usage statistics for CPU and formatted GPU monitoring
print(f"\nCPU Usage: {cpu_usage}%")
print(formatted_gpu_stats)

# Print summary
print("\n" + "=" * 72)
print(" " * 27 + "COMPLETE!")
print("=" * 72)
print(f"✅ Detected {len(faces)} face(s)")
print(f"✅ Output saved to: {output_path}")
print(f"✅ System monitoring complete")
print("=" * 72)

# Step 14: Display result (commented out for headless systems)
# cv2.imshow("Face Detection with Monitoring", img)
# cv2.waitKey(0)
cv2.destroyAllWindows()