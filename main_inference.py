from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import threading
import keyboard

# Global variable to control the pipeline's state
stop_pipeline = False

# Function to stop the pipeline when Enter key is pressed
def stop_pipeline_on_enter():
    global stop_pipeline
    print("Press Enter to stop...")
    keyboard.wait('enter')  # Waits for the Enter key press
    stop_pipeline = True

# Initialize the pipeline object
pipeline = InferencePipeline.init(
    model_id="rock-paper-scissors-sxsw/11",  # Roboflow model to use
    video_reference=0,  # Path to video, device id (int, usually 0 for built-in webcams), or RTSP stream URL
    on_prediction=render_boxes,  # Function to run after each prediction
)

# Start a separate thread to monitor for the Enter key press
stop_thread = threading.Thread(target=stop_pipeline_on_enter)
stop_thread.start()

# Start the pipeline and run until the stop flag is set
pipeline.start()

# Efficiently monitor the stop flag
try:
    while not stop_pipeline:
        pass  # Minimal busy-wait loop, just checking the flag

    # Stop the pipeline gracefully as soon as the flag is set
    pipeline.stop()

finally:
    # Ensure the stop thread is cleaned up
    stop_thread.join()

print("Pipeline stopped successfully.")



