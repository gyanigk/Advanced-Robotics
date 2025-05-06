import tkinter as tk
from tkinter import ttk
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import threading
from PIL import Image as PILImage, ImageTk

class Ros2TelemetryGUI(Node):
    def __init__(self, root):
        # Initialize ROS 2 node
        super().__init__('telemetry_gui')
        self.root = root
        self.root.title("ROS 2 Telemetry Dashboard")
        self.bridge = CvBridge()
        
        # Create GUI elements
        self.create_widgets()
        
        # Subscribe to ROS 2 topics
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10)
        
        # Initialize telemetry variables
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.camera_frame = None
        self.photo = None
        
        # Start update loop
        self.update_gui()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Telemetry display
        ttk.Label(main_frame, text="Telemetry Data", font=("Arial", 14)).grid(
            row=0, column=0, columnspan=2, pady=5)
        
        # Velocity labels
        self.linear_label = ttk.Label(main_frame, text="Linear Velocity: 0.0 m/s")
        self.linear_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        self.angular_label = ttk.Label(main_frame, text="Angular Velocity: 0.0 rad/s")
        self.angular_label.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Camera display
        self.canvas = tk.Canvas(main_frame, width=320, height=240)
        self.canvas.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Quit button
        ttk.Button(main_frame, text="Quit", command=self.quit).grid(
            row=4, column=0, columnspan=2, pady=5)

    def cmd_vel_callback(self, msg):
        self.linear_vel = msg.linear.x
        self.angular_vel = msg.angular.z

    def camera_callback(self, msg):
        try:
            # Convert ROS 2 image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Resize image for display
            cv_image = cv2.resize(cv_image, (320, 240))
            self.camera_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def update_gui(self):
        # Update velocity labels
        self.linear_label.config(text=f"Linear Velocity: {self.linear_vel:.2f} m/s")
        self.angular_label.config(text=f"Angular Velocity: {self.angular_vel:.2f} rad/s")
        
        # Update camera feed
        if self.camera_frame is not None:
            # Convert to PIL Image and then to PhotoImage for Tkinter
            pil_image = PILImage.fromarray(self.camera_frame)
            self.photo = ImageTk.PhotoImage(image=pil_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Schedule next update
        self.root.after(100, self.update_gui)

    def quit(self):
        self.destroy_node()
        rclpy.shutdown()
        self.root.quit()# Convert to HSV color space
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Define blue color range
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create mask for blue color
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

def main():
    # Initialize ROS 2
    rclpy.init()
    
    # Create Tkinter root
    root = tk.Tk()
    
    # Create GUI application
    app = Ros2TelemetryGUI(root)
    
    # Start ROS 2 spinning in a separate thread
    def ros_spin():
        rclpy.spin(app)
    
    ros_thread = threading.Thread(target=ros_spin, daemon=True)
    ros_thread.start()
    
    # Start Tkinter main loop
    root.mainloop()

if __name__ == '__main__':
    main()