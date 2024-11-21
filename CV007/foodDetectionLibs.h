#ifndef FOODDECT_LIBS_H
#define FOODDECT_LIBS_H
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <math.h>  
#include <limits>
#include <string>
#include "opencv2/imgproc.hpp"
#include <cmath>
#include <tuple>
#include "foodSegBoxLibs.h"


/**********************DETECTION FUNCTIONS******************************************/
std::vector<std::tuple<cv::Mat,cv::Point,double>> getFoodFromPlates(cv::Mat img);
std::vector<std::tuple<std::string,cv::Point,double,int>> foodDetectionWithColor(cv::Mat src);
std::vector<std::tuple<std::string,cv::Point,double,int>> leftoverDetection(cv::Mat src,std::vector<std::tuple<std::string,cv::Point,double,int>>listOfFoods,int flag);
std::map<std::string,double> firstCoursesDetection(cv::Mat src);
std::map<std::string,double> secondCoursesDetection(cv::Mat src);
std::map<std::string,double> sideDishesDetection(cv::Mat src);
std::map<std::string,double> getSamplesSaturation();
std::string getSideDish(cv::Mat secondCourseImg);


/********************UTILITY FUNCTIONS*****************************/

std::map<std::string,cv::Mat> readDataset();
cv::Mat houghTrasform(cv::Mat img);
cv::Mat extractRangeCOlorFromPlate(cv::Mat src);



#endif 