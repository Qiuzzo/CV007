#ifndef METRICS_LIBS_H
#define METRICS_LIBS_H
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
#include "foodDetectionLibs.h"
#include "foodSegBoxLibs.h"
#include <fstream>
#include <experimental/filesystem>
#include <sys/stat.h>



/*mIoU metrics
*function that compute the mIoU between two masks images given as parameter
*@param predictiomn the mask segmented by out algorithm
*@param the ground truth mask
*@return mIoU value
*/
double mIoU(cv::Mat prediction,cv::Mat gt);



/*food leftover
*function that compute the leftover % between two masks images given as parameter
*@param the mask of the before tray, segmented by our algorithm
*@param the mask of the leftover tray, segmented by our algorithm
*@return Ri value
*/
double foodLeftOverMetrics(cv::Mat before, cv::Mat after);

/*mAP
*function that compute the mAP metrics above all the trays in the Ground Truth
*@return mAP value
*/
double mAP();

/*Utility function used for mAP*/
std::vector<std::map<int,cv::Mat>> readGTboxes();

std::vector<cv::Mat> getTraysImages();

std::map<int,std::vector<std::string>> getFoodList();

std::map<int,cv::Mat> drawBoundingBoxLabel_For_metrics(std::vector<std::string> names, cv::Mat img);

bool comparePairs(const std::pair<int, int>& a, const std::pair<int, int>& b);

#endif 