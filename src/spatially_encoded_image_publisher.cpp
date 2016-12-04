#include <iostream>
#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/people/ground_based_people_detection_app.h>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>

#define PERSON_HALF_WIDTH 0.5
#define MIN_HEIGHT 1.3
#define MAX_HEIGHT 2.3
#define IMAGE_WIDTH 299
#define IMAGE_HEIGHT 299

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

sensor_msgs::CameraInfo camera_info_global;
bool is_camera_info_global = false;

ros::Publisher encodedImagePublisher;
ros::Publisher pclPublisher;

void extractPerson(PointCloudT::Ptr cloud)
{
  // Algorithm parameters:
  std::string svm_filename = ros::package::getPath(ROS_PACKAGE_NAME) + "/parameters/trainedLinearSVMForPeopleDetectionWithHOG.yaml";
  float min_confidence = -1.3;
  float min_height = 1.3;
  float max_height = 2.3;
  float voxel_size = 0.06;

  Eigen::VectorXf ground_coeffs(4);
  ground_coeffs.resize(4);
  ground_coeffs << 0.0239433, -0.999262, -0.0300222, 0.820499;

  Eigen::Matrix3f rgb_intrinsics_matrix;
  
  if (is_camera_info_global) {
    rgb_intrinsics_matrix << camera_info_global.K[0], camera_info_global.K[1], camera_info_global.K[2], camera_info_global.K[3], 
      camera_info_global.K[4], camera_info_global.K[5], camera_info_global.K[6], camera_info_global.K[7], camera_info_global.K[8];

    // std::cout << "Intrinsics: " << rgb_intrinsics_matrix << std::endl;
  } else {
    return;
  }

  // Create classifier for people detection:  
  pcl::people::PersonClassifier<pcl::RGB> person_classifier;
  person_classifier.loadSVMFromFile(svm_filename);   // load trained SVM

  // People detection app initialization:
  pcl::people::GroundBasedPeopleDetectionApp<PointT> people_detector;    // people detection object
  people_detector.setVoxelSize(voxel_size);                        // set the voxel size
  people_detector.setIntrinsics(rgb_intrinsics_matrix);            // set RGB camera intrinsic parameters
  people_detector.setClassifier(person_classifier);                // set person classifier
  people_detector.setHeightLimits(min_height, max_height);         // set person classifier
//  people_detector.setSensorPortraitOrientation(true);             // set sensor orientation to vertical

  // std::cout << "here" << std::endl;

  // Perform people detection on the new cloud:
  std::vector<pcl::people::PersonCluster<PointT> > clusters;   // vector containing persons clusters
  people_detector.setInputCloud(cloud);
  people_detector.setGround(ground_coeffs);                    // set floor coefficients
  people_detector.compute(clusters);                           // perform people detection

  ground_coeffs = people_detector.getGround();                 // get updated floor coefficients

  PointCloudT::Ptr no_ground_cloud (new PointCloudT);
  no_ground_cloud = people_detector.getNoGroundCloud();

  unsigned int k = 0;

  PointCloudT::Ptr cluster_cloud (new PointCloudT);
  PointCloudT::Ptr transformed_cloud (new PointCloudT);

  cv::Mat transformed_image( camera_info_global.height, camera_info_global.width, CV_8UC3, cv::Scalar::all(0) );

  for(std::vector<pcl::people::PersonCluster<PointT> >::iterator it = clusters.begin(); it != clusters.end(); ++it)
  {
    if(it->getPersonConfidence() > min_confidence)             // draw only people with confidence above a threshold
    {
      Eigen::Vector3f top = it->getTop();
      Eigen::Vector3f bottom = it->getBottom();

      Eigen::Vector4f minPoints;
      minPoints[0] = top[0]-PERSON_HALF_WIDTH;
      minPoints[1] = top[1];
      minPoints[2] = top[2]-PERSON_HALF_WIDTH;
      minPoints[3] = 1;

      Eigen::Vector4f maxPoints;
      maxPoints[0] = bottom[0]+PERSON_HALF_WIDTH;
      maxPoints[1] = bottom[1];
      maxPoints[2] = bottom[2]+PERSON_HALF_WIDTH;
      maxPoints[3] = 1;

      // std::cout << maxPoints << std::endl << " " << std::endl << minPoints << std::endl;

      pcl::CropBox<PointT> cropFilter;
      cropFilter.setInputCloud(cloud);
      cropFilter.setMin(minPoints);
      cropFilter.setMax(maxPoints);
      cropFilter.filter(*cluster_cloud);

      Eigen::Affine3f transform_origin = Eigen::Affine3f::Identity();
      transform_origin.translation() << -minPoints[0], -minPoints[1], -minPoints[2];
      pcl::transformPointCloud(*cluster_cloud, *transformed_cloud, transform_origin);

      // create new image with (r, g, b) = (x, y, z)
      float min_z = INFINITY;
      float min_x = INFINITY;
      float min_y = INFINITY;
      float max_x = -INFINITY;
      float max_y = -INFINITY;
      float max_z = -INFINITY;
      for (PointCloudT::iterator b1 = transformed_cloud->points.begin(); b1 < transformed_cloud->points.end(); b1++)
      {
        if (b1->x < min_x) {
          min_x = b1->x;
        }
        if (b1->x > max_x) {
          max_x = b1->x;
        }
        if (b1->y < min_y) {
          min_y = b1->y;
        }
        if (b1->y > max_y) {
          max_y = b1->y;
        }
        if (b1->z < min_z) {
          min_z = b1->z;
        }
        if (b1->z > max_z) {
          max_z = b1->z;
        }
      }

      int min_i = 1000;
      int min_j = 1000;
      int max_i = 0;
      int max_j = 0;
      for (PointCloudT::iterator b1 = cluster_cloud->points.begin(); b1 < cluster_cloud->points.end(); b1++)
      {
        // translated so that bottom is always origin
        float translated_x = b1->x - minPoints[0];
        float translated_y = b1->y - minPoints[1];
        float translated_z = b1->z - minPoints[2];

        // map the 3D coords back to 2D image coords  
        int i = round((b1->x * rgb_intrinsics_matrix(0, 0) / b1->z) + rgb_intrinsics_matrix(0, 2));
        int j = round((b1->y * rgb_intrinsics_matrix(1, 1) / b1->z) + rgb_intrinsics_matrix(1, 2));
        
        if (i >= transformed_image.cols) {
          i = transformed_image.cols - 1;
        }
        if (j >= transformed_image.rows - 1) {
          j = transformed_image.rows - 1;
        }

        if (i > max_i) {
          max_i = i;
        }
        if (j > max_j) {
          max_j = j;
        }
        if (i < min_i) {
          min_i = i;
        }
        if (j < min_j) {
          min_j = j;
        }
        transformed_image.at<cv::Vec3b>(j, i)[0] = (translated_x - min_x) / (2 * PERSON_HALF_WIDTH) * 255;
        transformed_image.at<cv::Vec3b>(j, i)[1] = (translated_y - min_y) / (MAX_HEIGHT) * 255;
        transformed_image.at<cv::Vec3b>(j, i)[2] = (translated_z - min_z) / (2 * PERSON_HALF_WIDTH) * 255;
      }

      cv::Mat cropped_image(transformed_image, cv::Rect(min_i, min_j, max_i-min_i, max_j-min_j));
      cv::Mat resized_image;
      cv::resize(cropped_image, resized_image, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));

      // publish image
      sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", resized_image).toImageMsg();
      pub.publish(msg);

      // PointCloudT::Ptr voxel_filtered (new PointCloudT ());

      // // Create the filtering object
      // pcl::VoxelGrid<PointT> sor;
      // sor.setInputCloud (transformed_cloud);
      // sor.setLeafSize (0.01f, 0.01f, 0.01f);
      // sor.filter (*voxel_filtered);

      // std::cerr << "PointCloud after filtering: " << voxel_filtered->width * voxel_filtered->height 
      //  << " data points (" << pcl::getFieldsList (*voxel_filtered) << ").";

//      pcl::PCDWriter writer;
//         writer.write ("table_scene_lms400_downsampled.pcd", *voxel_filtered, 
//         Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), false);

      // pcl::io::savePCDFileASCII ("/home/rakesh/person.pcd", *voxel_filtered);

      // Convert to ROS data type
      // pub.publish (*voxel_filtered);

/*
      // draw theoretical person bounding box in the PCL viewer:
      pcl::PointIndices clusterIndices = it->getIndices();    // cluster indices
      std::vector<int> indices = clusterIndices.indices;
      for(unsigned int i = 0; i < indices.size(); i++)        // fill cluster cloud
      {
        PointT* p = &no_ground_cloud->points[indices[i]];
        cluster_cloud->push_back(*p);
      }
*/

      k++;
    }
    std::cout << "Found " << k << " persons" << std::endl;
  }
}

void imageCallback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info_msg)
{
  camera_info_global = *info_msg;
  is_camera_info_global = true;
}

void pclCallback(const PointCloudT::ConstPtr& pcl_msg)
{
  std::cout << "PCL: " << pcl_msg->width << "X" << pcl_msg->height << std::endl;
  std::cout << std::endl;

  PointCloudT::Ptr cloud (new PointCloudT( *pcl_msg ) );
  extractPerson( cloud );
  
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;
  // cv::namedWindow("view");
  // cv::startWindowThread();
  image_transport::ImageTransport it(nh);
  

  // Publish the data
  // pub = nh.advertise<sensor_msgs::PointCloud2> ("output", 1);
  encodedImagePubisher = nh.advertise<sensor_msgs::Image> ("/person", 1);
  pclPubisher = nh.advertise<sensor_msgs::Image> ("/points", 1);
  
  // Subscribe data (one)
  ros::Subscriber sub = nh.subscribe<PointCloudT>("/camera/depth_registered/points", 1, pclCallback);
  // Subscribe data (simultaneous)
  message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "camera/rgb/image_raw", 1);
  message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub(nh, "camera/rgb/camera_info", 1);
  message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::CameraInfo> sync(image_sub, info_sub, 10);
  sync.registerCallback(boost::bind(&imageCallback, _1, _2));

  ros::spin();
  // cv::destroyWindow("view");
}
