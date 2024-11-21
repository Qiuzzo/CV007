

#ifndef FOODSEGBOX_LIBS_H
#define FOODSEGBOX_LIBS_H
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



/**********************BOUNDING BOX*******************************/
void getTrayWithOnlyFood(cv::Mat img, std::vector<cv::Point>& centers,std::vector<int>& radiuses,std::vector<cv::Mat>& plates);
std::vector<std::vector<cv::Rect>> computeBoundingBox(cv::Mat img);
cv::Mat drawBoundingBox(cv::Mat img);
cv::Mat drawBoundingBoxLabel(std::vector<std::string> names, cv::Mat img);
bool copyImg(cv::Mat img, std::vector<cv::Mat>& plates, cv::Point center, int radius);



/********************FOOD_SEGMENTATION****************************/
std::vector<std::tuple<cv::Mat, cv::Point>> foodSegmentation(cv::Mat src, std::vector<std::tuple<std::string, cv::Point, double, int>> platesWithNames);
static void cannyThreshold(int, void* userData);
void wBackground(cv::Mat& src, cv::Mat& dest);
void wForeground(cv::Mat& src, cv::Mat& dest, cv::Mat& fg);
void wMarkers(cv::Mat& bg, cv::Mat& markers, std::vector<std::vector<cv::Point>>& contours);
cv::Mat randomColors(cv::Mat& markers, std::vector<cv::Vec3b>& colors, std::vector<std::vector<cv::Point>> contours);
std::vector<cv::Mat> applyWatershed(cv::Mat src, cv::Mat it);
cv::Mat M_grabCut(cv::Mat& img,std::vector<cv::Point>& centers, std::vector<int>& radiuses);
void color(cv::Mat& singlePlate, const cv::Vec3b& color);
cv::Mat entireMask(std::vector<std::tuple<cv::Mat, cv::Point>> masks, cv::Mat img, std::vector<std::tuple<std::string,cv::Point,double, int>> label);


//DA ELIMINARE
cv::Mat foodSegmentationUtil(cv::Mat src, std::vector<std::tuple<std::string, cv::Point, double, int>> platesWithNames);


#endif 
