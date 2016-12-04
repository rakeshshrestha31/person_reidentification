#!/usr/bin/env python
import roslib
import sys
import os
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import rospkg
import openface
#from __future__ import print_function

roslib.load_manifest('person_reidentification')

class image_converter:
  def __init__(self):
    rospack = rospkg.RosPack()
    nodePath = rospack.get_path('person_reidentification')
    networkModelPath = os.path.join(nodePath, 'model', 'nn4.small2.t7')

    self.net = openface.TorchNeuralNet(networkModelPath, 96, True)
    self.dictionary = {}
    self.index = 1

#    self.image_pub = rospy.Publisher("",Image, queue_size=1)

    self.bridge = CvBridge()
    
    self.image_sub = message_filters.Subscriber('/person', Image, callback)
    ts.registerCallback(self.callback)

  def callback(self, image):
    try:
        cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
    except CvBridgeError as e:
        print(e)
    
    (rows,cols,channels) = cv_image.shape
    resized_image = cv2.resize(cv_image, (96, 96))
    rep = getRep(resized_image)

    min_distance = 0.5
    min_key = -1
    for key, reps in self.dictionary.iteritems():
       distance = self.getMinL2Dictionary(dictionary, rep)
       if distance < min_distance:
          min_distance = distance
          min_key = key

    if min_key == -1:
       dictionary[self.index] = []
       dictionary[self.index].append(rep)
       self.index ++
       print "Create new item, {}".format(self.index)
    else:
       dictionary[min_key].append(rep)
       print "Found the person {} with distance {}".format(min_key, min_distance)

  def getRep(image):
    rep = self.net.forward(image)
    return rep

  def getMinL2Distance(self, vector_list_a, vector_b):
    return min([np.dot(vector_a - vector_b, vector_a - vector_b) for vector_a in vector_list_a])

def main(args):
  ic = image_converter()
  rospy.init_node('person_reidentification', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
