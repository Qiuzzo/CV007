/*
* @author Carlotta Schiavo,ID=2076743 and Qiu yi jian, 	ID=2085730
*/
#include "foodSegBoxLibs.h"
#include "foodDetectionLibs.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


/* 
* function getTrayWithOnlyFood:
* This function computes the HoughCircle on the image of the tray and modify the two input parameters adding the center and the radius
* of each circle
* @param img: original image
* @param centers: vector in which will be stored the centers of the plates
* @param radiuses: vector in which will be stored the radiuses of the circle found by HoughCircle
*/
void getTrayWithOnlyFood(Mat img, vector<Point>& centers,vector<int>& radiuses,vector<Mat>& plates){
    cv::Mat gray;
    Mat cloned_img=img.clone();

    cvtColor( cloned_img, gray, cv::COLOR_BGR2GRAY );

    //apply a gaussian filter for the HoughCircles function
    cv::GaussianBlur( gray, gray, cv::Size(7,7), 0 );

    //vector that stores alle the circles identified by HoughCircles
    std::vector<cv::Vec3f> circles;
    HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, 420, 200, 50, 0, 0);

    for( size_t i = 0; i < circles.size(); i++ ){
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        centers.push_back(center);
        radiuses.push_back(radius);

        //check if the circle highligthed a glass, if so, it will be removed from the two vectors
        bool glass = copyImg(img, plates, center, radius);
       
        if (glass == 1){
            centers.pop_back();
            radiuses.pop_back();
           
        }
    }
}

/* 
* function copyImg:
* This function takes in input an image and after checking the color ratio of the pixels in the circles and also the cricles area,
* it gives in output a bool that tells if a circles contains just a glass or it's a plate, if the circle contains a plate,
* the plate will be added to the vector plates
* @param img: original image
* @param plates: vector in which will be stored all the circles with food
* @param center: the center of the circles that the function will analyze
* @param radius: radius of the circle
* @return true if the image is a glass
*/
bool copyImg(Mat img, vector<Mat>& plates, Point center, int radius){
	Mat copy = Mat::zeros(img.rows, img.cols, img.type());
	
    bool glass = 1;
    double Threshold = 0.3;
 	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols; j++){
			int dist = sqrt((pow((center.x - j),2) + pow((center.y - i),2)));
			if (dist < radius) {
				copy.at<cv::Vec3b>(i,j)[0] = img.at<Vec3b>(i, j)[0];
				copy.at<cv::Vec3b>(i,j)[1] = img.at<Vec3b>(i, j)[1];
				copy.at<cv::Vec3b>(i,j)[2] = img.at<Vec3b>(i, j)[2];
                double valA = static_cast<double>(copy.at<cv::Vec3b>(i,j)[0]);
				double valB = static_cast<double>(copy.at<cv::Vec3b>(i,j)[1]);
				double valC = static_cast<double>(copy.at<cv::Vec3b>(i,j)[2]);
				//cout <<"Convertito ["<< i << "," << j << "]" <<endl;
				double d1 = abs((static_cast<double>(valA/valB))-1);//abs(original_img.at<cv::Vec3b>(i,j)[0] - 200);
         	    double d2 = abs((static_cast<double>(valC/valB))-1);//abs(original_img.at<cv::Vec3b>(i,j)[1] - 200);
         	    double d3 = abs((static_cast<double>(valA/valC))-1);//abs(original_img.at<cv::Vec3b>(i,j)[2] - 200);
				double area = M_PI*pow(radius,2);
                if((d1 > Threshold || d2 > Threshold || d3 > Threshold) && area>80000 ){
					glass = 0;
				}
			}
			else{
				copy.at<cv::Vec3b>(i,j)[0] = 0;
				copy.at<cv::Vec3b>(i,j)[1] = 0;
				copy.at<cv::Vec3b>(i,j)[2] = 0;
			}
		}
	}
	if(glass==0)
	    plates.push_back(copy);
	
    return glass;
}

/* 
* function computeBoundingBox:
* This function takes in input the tray and will compute the bounding box of the food, getting the plates with food from the
* function getTrayWithOnlyFood
* @param img: image of the tray
* @return: the function returns a vector of bounding box that are stored in a vector con Rect
*/
vector<vector<Rect>> computeBoundingBox(Mat img){
     
    vector<Point> centers;
    vector<int> radiuses;
    vector<Mat> platesWhite;
    vector<Mat> plates;
    getTrayWithOnlyFood(img,centers,radiuses,platesWhite);

    vector<vector<Rect>> boxes;

    for(int i=0;i<platesWhite.size();i++){
        Mat it=M_grabCut(platesWhite[i], centers, radiuses);
        plates.push_back(it);
    }

    for(Mat& it : plates){
            Mat gray;
            cvtColor(it, gray, COLOR_BGR2GRAY);

            vector<vector<Point> > contours;
            findContours(gray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);    

            vector<vector<Point> > contours_poly( contours.size() );
            vector<Rect> boundRect( contours.size() );
           
            for( int i = 0; i < contours.size(); i++ ){
                if(contourArea(contours[i]) > 4800){
                    approxPolyDP( contours[i], contours_poly[i], 3, true );
                    boundRect[i] = boundingRect( contours_poly[i]);
                }
            }
            boxes.push_back(boundRect);
	    }
    return boxes;
}

/* 
* function drawBoundingBox:
* This function takes in input the tray and will draw the bounding box in the image, using at the beginning the function computeBoundingBox
* to computer the bounding box.
* @param img: image of the tray
* @return: the function returns the image with all the bounding box
*/
Mat drawBoundingBox(Mat img){

    vector<vector<Rect>> boxList=computeBoundingBox(img);
    
    for( int i = 0; i < boxList.size(); i++ ){
        for(int j=0;j<boxList[i].size();j++){
            rectangle(img, boxList[i][j].tl(), boxList[i][j].br(), Scalar(0, 0, 255), 2 );
        }
    }
    return img;
}

/* 
* function drawBoundingBoxLabel:
* This function takes in input the tray and a vector with the name of the foods 
* and will print the name of the food with the bounding box,
* @param names: names of the food in the image tray
* @param img: image of the tray
* @return: the function returns the image with all the bounding box and the label of the foods
*/
Mat drawBoundingBoxLabel(vector<string> names, Mat img){
    vector<vector<Rect>> boxList=computeBoundingBox(img);
   
        const vector<Vec3b> colors = {
            Vec3b(0, 0, 255),
            Vec3b(0, 255, 0),
            Vec3b(255, 0, 0),
            Vec3b(100, 100, 100),
            Vec3b(255, 255, 0)
        };
   
    
        for( int i = 0; i < boxList.size(); i++ )
        {
            for(int j=0;j<boxList[i].size();j++){
                rectangle(img, boxList[i][j].tl(), boxList[i][j].br(), colors[i], 2 );
                if(boxList[i][j].tl().x!=0 &&boxList[i][j].tl().x!=0){
                    Point up_corner=boxList[i][j].tl();
                    up_corner.y+=20;
                    //write the labels
                    putText(img, names[i],up_corner,cv::FONT_HERSHEY_SIMPLEX,1,  colors[i],2);
                }
            }
        }
        return img;
}

/*
* Watershed function (not working). The objective of this function was to divide the two or three food in the same plate
* The idea was to find the region with watershed and the analyze each reagion and using the color of each area, to define if it
* belongs to a certain class (food) or not and returning the mask of the food
*/
vector<Mat> applyWatershed(Mat src, Mat it){
    Mat clone = Mat::zeros(src.size(), src.type());
    for(int i = 0; i<src.rows; i++){
        for(int j = 0; j<src.cols; j++){
            clone.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0];
            clone.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1];
            clone.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2];
        }
    }

    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));

    morphologyEx(it, it, MORPH_DILATE, morphKernel, Point(-1, -1), 2);
    Mat kernel = (Mat_<float>(3,3) <<
                    1, 1, 1,
                    1, -8, 1,
                    1, 1, 1);

    //apply Laplacian
    Mat imgLaplacian;
    filter2D(it, imgLaplacian, CV_32F, kernel);
    Mat sharp;
    it.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;

    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

    //create binary from source
    Mat bw;
    cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
    //imshow("Binary Image", bw);

    Mat dist;
    distanceTransform(bw, dist, DIST_L2, 3);
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    //imshow("Distance Transform Image", dist);
    threshold(dist, dist, 0.4, 1.0, THRESH_BINARY); //peak as foreground objects
    Mat kernel1 = Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1);
    //imshow("Peaks", dist);
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    //find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat markers = Mat::zeros(dist.size(), CV_32S);
    //draw the foreground makers that are not regions
    for (size_t i = 0; i < contours.size(); i++){
        drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i)+1), -1);
    }

    //draw background marker
    circle(markers, Point(5,5), 3, Scalar(255), -1);
    Mat markers8u;
    markers.convertTo(markers8u, CV_8U, 10);
    //imshow("Markers", markers8u);
    watershed(imgResult, markers); //WATERSHED ALGORITHM

    Mat seed; 
    markers.convertTo(seed, CV_8U);
    imshow("Markers", seed);


    Mat mark;
    markers.convertTo(mark, CV_8U);
    bitwise_not(mark, mark);

    Mat labels;
    int numRegions = connectedComponents(mark, labels);

    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++){
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++){
        for (int j = 0; j < markers.cols; j++){
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size())){
                dst.at<Vec3b>(i,j) = colors[index-1];
                //cout << "INDEX" << index << endl;
            }
        }
    }

    // Visualize the final image
    imshow("Final Result", dst);

    vector<vector<Point> > conRegion;
    vector<Vec4i> hierarchy;
    cvtColor(dst,dst, COLOR_BGR2GRAY);
    findContours(dst, conRegion, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    //Mat d;
    //drawContours(d, conRegion, -1, Scalar(255,255,255), 1, 8);
    Mat drawing = Mat::zeros( dst.size(), CV_8UC3 );
    RNG rng(1);
    vector<Mat> regioni;
        
    for( size_t i = 0; i< conRegion.size(); i++ ){
        Mat temp = Mat::zeros(dst.size(), CV_8U);
        //Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( temp, conRegion, (int)i, Scalar(255,255,255), 2, LINE_4, hierarchy, 0 );
        //cout<<"SONO QUI"<<endl;
        regioni.push_back(temp);
    }

    for(Mat i:regioni){
        imshow("SINGLE REGION", i);
        waitKey(0);
    }
    vector<Mat> cibi;


    return cibi;
}

/*
* funciotn foodSegmentation
* This function takes in input the tray image and the names of the plates with the respective center and radius with the identifier code
* It's objective is to segment the food of each plate using the function getTrayWithOnlyFood and to color the segmented food with a color
* @param src: tray image
* @param platesWithNames: vector with the name, the center, radius and the id code of the food on the plate
* @return: it returns a vector of masks in which each mask is associated with the id code of the food, this will be used to create the particular mask
*/
vector<tuple<Mat, Point>> foodSegmentation(Mat src, vector<tuple<string, Point, double, int>> platesWithNames){
    
    vector<tuple <Mat, Point>> masks;
    const vector<Vec3b> colors = {
		    Vec3b(0, 0, 255),
		    Vec3b(0, 255, 0),
		    Vec3b(255, 0, 0),
		    Vec3b(100, 100, 100),
            Vec3b(255, 255, 0)
	    };
    
    vector<Point> centers;
    vector<int> radiuses;
    vector<Mat> platesWhite;
    vector<Mat> plates;
    getTrayWithOnlyFood(src,centers,radiuses,platesWhite);
    
    Mat img = Mat::zeros(src.rows, src.cols, CV_8UC3);

    Mat morphKernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    for(int i=0;i<platesWhite.size();i++){
        Mat it=M_grabCut(platesWhite[i], centers, radiuses);
        morphologyEx(it, it, MORPH_CLOSE, morphKernel, Point(-1, -1), 2);
        
        plates.push_back(it); //saves the image with single plate in plates       
        color(plates[i], colors[i]); //color each segmented food with a particular color
    }

    //this for loop will take each image and using the grabCut function it will remove the plate leaving just the image with the food
    int scan = 0;
    for(Mat& it : platesWhite){
        it=M_grabCut(platesWhite[scan], centers, radiuses);
        morphologyEx(it, it, MORPH_CLOSE, morphKernel, Point(-1, -1), 2);
        tuple<Mat, Point> pair;
        get<0>(pair) = it;
        get<1>(pair) = centers[scan];
        masks.push_back(pair);
        scan++;
	}
    //this funtion colors the segmented food in the black image black image 
    for (Mat& it: plates){
        for (int i = 0; i < it.rows; i++){
		    for (int j = 0; j < it.cols; j++){
			    if (it.at<Vec3b>(i, j)[0] != 0 || it.at<Vec3b>(i, j)[1] != 0 || it.at<Vec3b>(i, j)[2] != 0){
			        img.at<Vec3b>(i, j)[0] = it.at<Vec3b>(i, j)[0];
			        img.at<Vec3b>(i, j)[1] = it.at<Vec3b>(i, j)[1];
			        img.at<Vec3b>(i, j)[2] = it.at<Vec3b>(i, j)[2];
			    }
            }
	    }
    }

  
  
    
    
    return masks;
}


/*
* function foodSegmentationUtil
* This function would highlight the segmnetation of the food, by coloring them using random colors
* @param src: src image 
* @param platesWithNames: list of detected plates
* @return: the function returns the image with all the mask where each region is colored with a random color
*/
Mat foodSegmentationUtil(Mat src, vector<tuple<string, Point, double, int>> platesWithNames){

     vector<tuple <Mat, Point>> masks;
    const vector<Vec3b> colors = {
		    Vec3b(0, 0, 255),
		    Vec3b(0, 255, 0),
		    Vec3b(255, 0, 0),
		    Vec3b(100, 100, 100),
            Vec3b(255, 255, 0),
            Vec3b( 127,	255,212	),
            Vec3b( 255,0,255),
            Vec3b( 0,255,255),
            Vec3b( 128,	255,255	),
            Vec3b( 255,	255,128	),
            Vec3b( 244,	12,	161	),
            Vec3b( 128	,0,	128	)
           
	    };
    
    vector<Point> centers;
    vector<int> radiuses;
    vector<Mat> platesWhite;
    vector<Mat> plates;
    getTrayWithOnlyFood(src,centers,radiuses,platesWhite);
    
    Mat img = Mat::zeros(src.rows, src.cols, CV_8UC3);

    Mat morphKernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    for(int i=0;i<platesWhite.size();i++){
        Mat it=M_grabCut(platesWhite[i], centers, radiuses);
        morphologyEx(it, it, MORPH_CLOSE, morphKernel, Point(-1, -1), 2);
        
        plates.push_back(it); //saves the image with single plate in plates       
        color(plates[i], colors[i]); //color each segmented food with a particular color
    }

    //this for loop will take each image and using the grabCut function it will remove the plate leaving just the image with the food
    int scan = 0;
    for(Mat& it : platesWhite){
        it=M_grabCut(platesWhite[scan], centers, radiuses);
        morphologyEx(it, it, MORPH_CLOSE, morphKernel, Point(-1, -1), 2);
        tuple<Mat, Point> pair;
        get<0>(pair) = it;
        get<1>(pair) = centers[scan];
        masks.push_back(pair);
        scan++;
	}
    //this funtion colors the segmented food in the black image black image 
    for (Mat& it: plates){
        for (int i = 0; i < it.rows; i++){
		    for (int j = 0; j < it.cols; j++){
			    if (it.at<Vec3b>(i, j)[0] != 0 || it.at<Vec3b>(i, j)[1] != 0 || it.at<Vec3b>(i, j)[2] != 0){
			        img.at<Vec3b>(i, j)[0] = it.at<Vec3b>(i, j)[0];
			        img.at<Vec3b>(i, j)[1] = it.at<Vec3b>(i, j)[1];
			        img.at<Vec3b>(i, j)[2] = it.at<Vec3b>(i, j)[2];
			    }
            }
	    }
    }

 
   
    Mat image=img.clone();
    resize(image, image, Size(image.cols/3, image.rows/3)); // to half size or even smaller


    return image;
    
}

/*
* funciotn entireMask
* This function with this input will take each mask of single food and color "maschera" with the corrisponding pixels using the id
* as color associated to a particular plate
* @param masks: vector of mask where each image is a mask of a single food
* @param img: sorce image used to take the dimentsion of output image
* @param label: vector with the name, center, radius and id of each detected food
* @return: the function returns the image with all the mask where each region is colored with the respective id
*/
Mat entireMask(vector<tuple<Mat, Point>> masks ,Mat img, vector<tuple<string,Point,double,int>> label){
    Mat maschera = Mat::zeros(img.rows, img.cols, CV_8U);
    for (tuple<Mat, Point> it:masks){
        Mat temp = get<0>(it);
        int colorMask = 0;
        for (tuple<string,Point,double,int> thing:label){

            if (get<1>(it).x == get<1>(thing).x && get<1>(it).y == get<1>(thing).y){
                colorMask = get<3>(thing);
                break;
            }
        }
       
        for (int i = 0; i < temp.rows; i++){
            for (int j = 0; j < temp.cols; j++){
                if (temp.at<Vec3b>(i,j)[0] != 0 ||
                   temp.at<Vec3b>(i,j)[1] != 0 ||
                   temp.at<Vec3b>(i,j)[2] != 0){
                   maschera.at<uchar>(i,j) = colorMask;
                }
            }
        }
    }
    return maschera;
}

/*
* function color
* This function color each plate with one of the color to highlight the segmentation
* @param singlePlate: image of the segmented food that will be colored using the color in "color"
* @param color: variable that contains the color used for the food in singlePlate
* @return
*/
void color(Mat& singlePlate, const Vec3b& color){

	// Convert the img to the HSV color space
    Mat hsvImage;
    cvtColor(singlePlate, hsvImage, COLOR_BGR2HSV);

    // Extract the saturation channel
    Mat satChannel;
    extractChannel(hsvImage, satChannel, 1);

    // Reshape the img to a 2D matrix of pixels
    Mat reshapedImage = satChannel.reshape(1, satChannel.rows * satChannel.cols);

    // Convert the reshaped img to a float type
    reshapedImage.convertTo(reshapedImage, CV_32F);

    // Perform k-means clustering with k = 2
    Mat labels, centers1;
    kmeans(reshapedImage, 2, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_RANDOM_CENTERS, centers1);

    // Assign each pixel to its corresponding cluster center
    Mat segmented = Mat::zeros(reshapedImage.size(), reshapedImage.type());
    for (int i = 0; i < reshapedImage.rows; ++i) {
        for (int j = 0; j < reshapedImage.cols; ++j) {
            int clusterIdx = labels.at<int>(i * reshapedImage.cols + j);
            segmented.at<float>(i, j) = centers1.at<float>(clusterIdx);
        }
    }

    // Convert the segmented img back to the original shape
    segmented = segmented.reshape(0, satChannel.rows);

    // Normalize the segmented img to the range [0, 255]
    normalize(segmented, segmented, 0, 255, NORM_MINMAX, CV_8U);

	segmented.convertTo(segmented, CV_8U);

    for (int i = 0; i < segmented.rows; i++){
        for (int j = 0; j < segmented.cols; j++){
            if (segmented.at<uchar>(i, j) == 255){
                singlePlate.at<Vec3b>(i, j)[0] = color[0];
                singlePlate.at<Vec3b>(i, j)[1] = color[1];
                singlePlate.at<Vec3b>(i, j)[2] = color[2];
            }
        }
    }
}

/*
* function M_grabCut
* This function at the beginning, colors all the pixel that respect a particular condition with a color which is similar to the color of the plate.
* The condition is that the ratio of the color of the pixel is around 1, so this means that the pixel is gray and this allows us to remove
* the pixels that are not food. Then the function create smaller image with the segmented food that will be used in other function.
* At the end, the grabCut function is used to highlight the food on the dish
* @param img: source image
* @param centers: vector of centers of the plates found by HoughCircles
* @param radiuses: vector of radiuses of the plates found by HoughCircles
* @return the image after the grabCut function
*/
Mat M_grabCut(Mat& img, vector<Point>& centers, vector<int>& radiuses){
    Mat clone, img2;

    img.copyTo(img2);
    img.copyTo(clone);

    double Threshold = 0.3;
    for (int i = 0; i < img.rows; i++){
        for (int j = 0; j < img.cols; j++){
            int scan = 0;
            bool out = 1;
            for (Point p: centers){
                int dist = sqrt((pow((p.x - j),2) +pow((p.y - i),2)));
                if (dist < radiuses.at(scan)){
                    out = 0;
                }
                scan++;
            }
            if (out == 1){
                clone.at<cv::Vec3b>(i,j)[0] = 200;
                clone.at<cv::Vec3b>(i,j)[1] = 200;
                clone.at<cv::Vec3b>(i,j)[2] = 200;
            } else{
				double valA = static_cast<double>(clone.at<cv::Vec3b>(i,j)[0]);
				double valB = static_cast<double>(clone.at<cv::Vec3b>(i,j)[1]);
				double valC = static_cast<double>(clone.at<cv::Vec3b>(i,j)[2]);
				//cout <<"Convertito ["<< i << "," << j << "]" <<endl;
				double d1 = abs((static_cast<double>(valA/valB))-1);//abs(original_img.at<cv::Vec3b>(i,j)[0] - 200);
         	    double d2 = abs((static_cast<double>(valC/valB))-1);//abs(original_img.at<cv::Vec3b>(i,j)[1] - 200);
         	    double d3 = abs((static_cast<double>(valA/valC))-1);//abs(original_img.at<cv::Vec3b>(i,j)[2] - 200);
				if(d1 <= Threshold && d2 <= Threshold && d3 <= Threshold){
					clone.at<cv::Vec3b>(i,j)[0] = 200;
            	    clone.at<cv::Vec3b>(i,j)[1] = 200;
            	    clone.at<cv::Vec3b>(i,j)[2] = 200;
				}
			}
        }
    }

    vector<tuple<Rect, Mat>> images;
	vector<Mat>foodPlates;

    for(int i=0;i<centers.size();i++){
        Point center=centers[i];
        int radius = radiuses[i];
        //create a new image that has dimensione diameter*diameter
        Mat temp(radius*2,radius*2,CV_8UC3,Scalar(0,0,0));

		vector<cv::Point> contour;
		contour.push_back(Point((center.x)-(radius), (center.y)-(radius)));
		contour.push_back(Point((center.x)+(radius), (center.y)-(radius)));
		contour.push_back(Point((center.x)+(radius), (center.y)+(radius)));
		contour.push_back(Point((center.x)-(radius), (center.y)+(radius)));

        Rect box = boundingRect(contour);
        
        int x=0;
        for(int i=(center.y)-radius;(i<=(center.y)+radius)&&(x<temp.rows);i++){
            int y=0;
            for(int j=(center.x)-radius;(j<=(center.x)+radius)&&(y<temp.cols);j++){
                temp.at<Vec3b>(y,x)[0]=img.at<Vec3b>(i,j)[0];
                temp.at<Vec3b>(y,x)[1]=img.at<Vec3b>(i,j)[1];
                temp.at<Vec3b>(y,x)[2]=img.at<Vec3b>(i,j)[2];
                y++;
            }
            x++;
        }

        foodPlates.push_back(temp);

		Mat crop_img = temp;
		tuple <Rect, Mat> t;
        t = make_tuple(box, crop_img);
        images.push_back(t);
    }

    // grabCut
	Mat mask2 = Mat::zeros(img.size(), CV_8UC1);
    Mat fgModel = Mat::zeros(1, 65, CV_64FC1);
    Mat bgModel = Mat::zeros(1, 65, CV_64FC1);

    Mat dst1;

    for (tuple<Rect, Mat> t : images) {
        int x, y, w, h;
        x = get<0>(t).x;
        y = get<0>(t).y;
        w = get<0>(t).width;
        h = get<0>(t).height;

        rectangle(img2, Rect(x, y, w, h), Scalar(0, 0, 255), 2);
        grabCut(clone, mask2, Rect(x, y, w, h), bgModel, fgModel, 1, GC_INIT_WITH_RECT);
        Mat mask3 = (mask2 == 3);
        clone.copyTo(dst1, mask3);
    }
    return dst1;
}





