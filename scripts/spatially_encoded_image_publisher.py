#!/usr/bin/env python
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
#from __future__ import print_function

roslib.load_manifest('person_reidentification')


class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("person_reidentification/spatially_encoded_rgb_image",Image, queue_size=1)

    self.bridge = CvBridge()
    
    self.image_sub = message_filters.Subscriber('camera/rgb/image_raw', Image)
    self.info_sub = message_filters.Subscriber('/camera/rgb/camera_info', CameraInfo)
    self.pcl_sub = message_filters.Subscriber('/camera/depth_registered/points', PointCloud2)

    ts = message_filters.TimeSynchronizer([self.image_sub, self.info_sub, self.pcl_sub], 10)
    ts.registerCallback(self.callback)
    

  def callback(self,image, info):
    try:
        cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
    except CvBridgeError as e:
        print(e)
    
    print info
      
    

    (rows,cols,channels) = cv_image.shape
    if cols > 60 and rows > 60 :
      cv2.circle(cv_image, (50,50), 10, 255)

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
