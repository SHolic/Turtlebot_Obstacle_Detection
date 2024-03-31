# Import necessary ROS2 and OpenCV libraries
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import math
import numpy as np


class TurtlebotController(Node):
    def __init__(self):
        # Initialize the node with the name 'turtlebot_controller'
        super().__init__('turtlebot_controller')

        # Movement control setup
        # Create a publisher for sending velocity commands
        # This publisher will send messages of type Twist to the 'cmd_vel' topic, 
        # which is commonly used for controlling robot motion. The queue size of 10
        # ensures that up to 10 messages can be buffered for sending if necessary, 
        # managing the flow of commands under varying system loads.
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        # Initial state for the movement logic
        self.state = "move_forward"
        # Initialize obstacle height
        self.height = 0
        # Create a timer to periodically update the robot's state
        self.timer = self.create_timer(0.1, self.update_state)

        # Subscribe to the camera topic to receive image messages
        # Create a subscription to listen for messages on the '/camera/image_raw' topic,
        # using the Image message type. The 'image_callback' function is called for each new message,
        # with a queue size of 10 to buffer messages if they arrive faster than they can be processed
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()  # Initialize a CvBridge to convert ROS images to OpenCV format


    def update_state(self):
        msg = Twist()

        # If the robot has completed its turn or if no obstacle is detected, it moves forward.
        if self.height >= 350:
            self.state = "turn"
        else:
            self.state = "move_forward"

        # Set the velocity based on the current state
        if self.state == "move_forward":
            msg.linear.x = 0.2
            msg.angular.z = 0.0
        elif self.state == "turn":
            msg.linear.x = 0.0
            msg.angular.z = -0.2  # Clockwise rotation

        self.publisher_.publish(msg)  # Publish the velocity command


    def image_callback(self, msg):
        # This method is called with each new image message from the camera
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error('Failed to convert image: ' + str(e))
            return

        # Convert the image to HSV color space for easier color detection
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define the range of red color in HSV
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        # Create a mask for red color
        mask = cv2.inRange(hsv_image, lower_red, upper_red)

        # Define the range of wood color in HSV
        lower_wood = np.array([0, 40, 40])
        upper_wood = np.array([50, 255, 255])
        # Create a mask for wood color
        mask2 = cv2.inRange(hsv_image, lower_wood, upper_wood)

        mask = mask + mask2

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # If the largest obstacle is larger than a threshold, stop or turn
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(cv_image, [largest_contour], -1, (255, 0, 0), 3)
            # Get bounding box of the largest contour
            x, y, width, self.height = cv2.boundingRect(largest_contour)
            # Draw the bounding box as a rectangle on the image (optional)
            cv2.rectangle(cv_image, (x, y), (x + width, y + self.height), (0, 255, 0), 3)
            # Now you have the width of the largest contour
            print("Height of the largest contour:", self.height)
        else:
            self.height = 0

        # Display the OpenCV image in a window
        cv2.imshow("Camera Image", cv_image)
        cv2.imshow("Detected Obstacles", mask)
        cv2.waitKey(1)  # Wait a bit for the window to update


def main(args=None):
    rclpy.init(args=args)  # Initialize ROS2 Python client library
    turtlebot_controller = TurtlebotController()  # Create the Turtlebot controller node
    rclpy.spin(turtlebot_controller)  # Keep the node running and responsive
    # Cleanup before exiting
    turtlebot_controller.destroy_node()
    cv2.destroyAllWindows()  # Close the OpenCV window
    rclpy.shutdown()  # Shutdown ROS2 Python client library


if __name__ == '__main__':
    main()
