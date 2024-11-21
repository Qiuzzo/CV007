/*
* @author Carlotta Schiavo,ID=2076743 and Qiu yi jian, 	ID=2085730
*/

#include "foodDetectionLibs.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


//Name of the dataset
const string DATASET="../CV007/segmented_samples"; //folder that contains all samples use for food detection
const string FORMAT="*.jpg";



/*Struct that contains all the color ranges*/
struct foodColor{
   
    //First courses:

    //Pasta
    Scalar yellow_light_min = Scalar(0,70,156);
    Scalar yellow_light_max= Scalar(40,255,255);
        //with_pesto
            Scalar green_pesto_min= Scalar(26,65,20);
            Scalar green_pesto_max= Scalar(45,255,255);
        //with_tomato_sauce
            Scalar red_sauce_min= Scalar(0,150,20);
            Scalar red_sauce_max= Scalar(10,255,255);
         //with_meat_sauce
            Scalar meat_sauce_min= Scalar(0,60,20);
            Scalar meat_sauce_max= Scalar(10,255,255);
        //with_clams_and_mussels
            Scalar orange_min= Scalar(15,130,130);
            Scalar orange_max= Scalar(20,255,255);
            Scalar mussel_color_min= Scalar(10,148,13);
            Scalar mussel_color_max= Scalar(17,255,255);
        //pilaw_rice
            Scalar green_peas_min= Scalar(20,140,0);
            Scalar green_peas_max= Scalar(40,170,255);
            Scalar yellow_rice_min= Scalar(0,60,175);
            Scalar yellow_rice_max= Scalar(30,170,255);
            Scalar yellow_pepper_min= Scalar(14,195,0);
            Scalar yellow_pepper_max= Scalar(120,255,255);
            Scalar red_pepper_min= Scalar(0,185,150);
            Scalar red_pepper_max= Scalar(20,255,255);

    //Second Courses:

        //fish_cutlet
             Scalar orange_light_min= Scalar(0,144,148);
             Scalar orange_light_max= Scalar(18,255,255);
             
        //rabbit
             Scalar brown_min= Scalar(0,165,50);
             Scalar brown_max= Scalar(20,216,193);
        //fish_salad
             Scalar pink_piece_min= Scalar(0,91,0);
             Scalar pink_piece_max= Scalar(10,160,255);
             Scalar purple_piece_min= Scalar(0,106,0);
             Scalar purple_piece_max= Scalar(179,178,50);
             Scalar white_piece_min= Scalar(12,0,190);
             Scalar white_piece_max= Scalar(179,93,255);
        //grilled_pork
             Scalar white_meat_min= Scalar(10,48,110);
             Scalar white_meat_max= Scalar(16,155,255);
             Scalar mushrooms_min= Scalar(0,93,22);
             Scalar mushrooms_max= Scalar(11,220,77);
        
    //side dishes:

        //salad
             Scalar red_tomato_min= Scalar(0,150,80);
             Scalar red_tomato_max= Scalar(5,255,255);
             Scalar green_salad_min= Scalar(14,73,142);
             Scalar green_salad_max= Scalar(26,255,255);
             Scalar purple_salad_min= Scalar(160,155,0);
             Scalar purple_salad_max= Scalar(177,255,255);
             Scalar orange_carrot_min= Scalar(10,155,175);
             Scalar orange_carrot_max= Scalar(25,255,255);

        //beans        
             Scalar peal_color_min= Scalar(0,101,0);
             Scalar peal_color_max= Scalar(8,255,255);

        //bread
            Scalar brown_light_min= Scalar(0,42,22);
            Scalar brown_light_max= Scalar(30,130,255);
        //potatoes
            Scalar yellow_potatoes_min= Scalar(19,37,20);
            Scalar yellow_potatoes_max= Scalar(37,255,255);

           
};


/* 
* function isGLass:
* function that check for each plate found on the tray, if this one is a glass or not
* @param src, the image of the tray
* @param the plate, (image, center and radius)
* @return bool, true if the plate is a glass
*/
bool isAGlass(tuple<Mat,Point,double> plateAndLocation,Mat src){
   
    Mat img=src;
    Point center=get<1>(plateAndLocation);
    double radius=get<2>(plateAndLocation);

    Mat copy = Mat::zeros(img.rows, img.cols, img.type());
      int glass=1;
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
                    double d1 = abs((static_cast<double>(valA/valB))-1);
                    double d2 = abs((static_cast<double>(valC/valB))-1);
                    double d3 = abs((static_cast<double>(valA/valC))-1);
                    double area = M_PI*pow(radius,2);
                    if(d1 > Threshold && d2 > Threshold && d3 > Threshold && area>80000 ){
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
    return glass==1;
}



/* 
* function leftoverDetection:
* function that makes the food detetion on a leftover tray
* 
* "WARNING": this function is thought to be call only after the foodDetection made on the tray "before".
*
* @param src, the image of the tray
* @param listOfDetection, list of detection return by the detection on the tray "before"
* @param flag, 1= leftover1, 2=leftover2
* @return the list of the detection,a vector of :(foodName, center,radius,foodCategory)
*/
vector<tuple<string,Point,double,int>> leftoverDetection(Mat src,vector<tuple<string,Point,double,int>> listOfDetection,int flag){
   
    
    vector<tuple<string,Point,double,int>> listOfFoodsFounded;
    Mat cloned_img=src.clone();
    vector<tuple<Mat,Point,double>> plateAndLocation=getFoodFromPlates(src); 
    vector<Mat> plates; 
    vector<tuple<Point,double>>locations;
    string foodFound;


    map<string,int> foodClassMap{
        {"pasta_with_pesto",1},
        {"pasta_with_tomato_sauce",2},
        {"pasta_with_meat_sauce",3},
        {"pasta_with_clams_and_mussels",4},
        {"pilaw_rice",5},
        {"grilled_pork",6},
        {"fish_cutlet",7},
        {"rabbit",8},
        {"seafood_salad",9},
        {"beans",10},
        {"basil_potatoes",11},
        {"salad",12},
        {"bread",13}
    };

   
        
    if(flag!=1 && flag!=2){
        cout<<"Please, insert the kind of the leftover (1 or 2)";
        exit(-1);
    }

    if(flag==1){
       cout<<"Starting the leftover 1 foodDetection....Please, wait some seconds\n";
        vector<string>listFoods;
        for (auto const& plate : plateAndLocation){
            if(!isAGlass(plate,src)){
                plates.push_back(get<0>(plate));  
                locations.push_back(make_tuple(get<1>(plate),get<2>(plate)));
            }   
        }

        for(int s=0;s<plateAndLocation.size();s++){
            double minDist=10000;
            string foodMinDist="";
            Point p1=get<1>(plateAndLocation[s]);//take the center of the i^th plate on the leftover tray
            double radius=get<2>(plateAndLocation[s]);
            for (int g=0;g<listOfDetection.size();g++){
                Point p2=get<1>(listOfDetection[g]);//take the center of the i^th plate on the leftover tray
                double eucl_distance=sqrt(pow(p1.x - p2.x,2)+pow(p1.y - p2.y,2));
                if(eucl_distance<minDist){
                    minDist=eucl_distance;
                    foodMinDist=get<0>(listOfDetection[g]);
                }
            }
            int id_food;
            if(foodMinDist.find("grilled_pork")!=string::npos){
                id_food=foodClassMap["grilled_pork"];
            }
            else if(foodMinDist.find("rabbit")!=string::npos){
                id_food=foodClassMap["rabbit"];
            }
            else if(foodMinDist.find("fish_cutlet")!=string::npos){
                id_food=foodClassMap["fish_cutlet"];
            }
            else if(foodMinDist.find("seafood_salad")!=string::npos){
                id_food=foodClassMap["seafood_salad"];
            }
            else{
                 id_food=foodClassMap[foodMinDist];
            }
            listOfFoodsFounded.push_back(make_tuple(foodMinDist,p1,radius,id_food));
            listFoods.push_back(foodMinDist);
        }
        Mat boxes=drawBoundingBoxLabel(listFoods,cloned_img);

        cout<<"Plate leftover founded:"<<endl;
        
        for (int w=0;w<listOfFoodsFounded.size();w++){
        
            cout<<get<0>(listOfFoodsFounded[w])<<endl;  
        } 

        Mat image=boxes.clone();
        resize(image, image, Size(image.cols/3, image.rows/3)); 
        /*string pathName="../CV007/ImgReport/boundingBoxDetectionImg/tray3/leftover1"; 
        imwrite(pathName+".png",image);*/
        namedWindow( "LeftOver Detection",cv::WINDOW_AUTOSIZE);
        imshow("LeftOver Detection",image);
        cout<<"Please, press a keyboard key to continue. Then wait some seconds..."<<endl;
        waitKey();

    }



    else{
       
       
        cout<<"Starting the leftover 2 foodDetection....Please, wait some seconds\n";
        for (auto const& plate : plateAndLocation){
            if(!isAGlass(plate,src)){
                plates.push_back(get<0>(plate));  
                locations.push_back(make_tuple(get<1>(plate),get<2>(plate)));
            }   
        }
    
        map<string,double> sampleSaturations=getSamplesSaturation();
        vector<pair<double,string>>  filteredSamplesSaturations;
        vector<string> listOfFoods;
        string side_dish_name="";

        for(int d=0;d<listOfDetection.size();d++){
            listOfFoods.push_back(get<0>(listOfDetection[d]));
        }
        //Filtering Saturations
        map<string,vector<double>> saturations_deltas_foreach_food;
        for(int f=0;f<listOfFoods.size();f++){
            string food_name=listOfFoods[f];
        
            if(food_name.find("beans")!=string::npos )
                side_dish_name="beans";
            else{
                if(food_name.find("basil_potatoes")!=string::npos){
                    side_dish_name="basil_potatoes";
                }
            } 
            
            if(food_name.find("salad")!=string::npos){
                    if(food_name.find("seafood_salad")!=string::npos){
                        filteredSamplesSaturations.push_back({sampleSaturations[food_name],food_name});
                    }
                    else{
                        filteredSamplesSaturations.push_back({sampleSaturations[food_name+"1"],food_name});
                        filteredSamplesSaturations.push_back({sampleSaturations[food_name+"2"],food_name});
                        filteredSamplesSaturations.push_back({sampleSaturations[food_name+"3"],food_name});
                        filteredSamplesSaturations.push_back({sampleSaturations[food_name+"4"],food_name});
                    }     
            }
            else{
                if(food_name.find("fish_cutlet")!=string::npos || food_name.find("pasta_with_clams_and_mussels")!=string::npos ){
                    filteredSamplesSaturations.push_back({sampleSaturations[food_name+"1"],food_name});
                    filteredSamplesSaturations.push_back({sampleSaturations[food_name+"2"],food_name});
                    filteredSamplesSaturations.push_back({sampleSaturations[food_name+"3"],food_name});
                }
                else{
                    if(food_name.find("grilled_pork")!=string::npos || food_name.find("rabbit")!=string::npos || food_name.find("pasta_with_tomato_sauce")!=string::npos){
                        filteredSamplesSaturations.push_back({sampleSaturations[food_name+"1"],food_name});
                        filteredSamplesSaturations.push_back({sampleSaturations[food_name+"2"],food_name});
                    }
                    else{
                            filteredSamplesSaturations.push_back({sampleSaturations[food_name],food_name});
                    }
                }
            }
            vector<double> deltas; //ex. [delatSat1,deltaSat2,deltaSat3]...
            for(int i=0;i<plates.size();i++){

                vector<pair<double,string>> percAndPlate;
                double plate_saturation;
                Mat plateImgHSV,shifted,dst;
                foodFound="";
            
                cv::pyrMeanShiftFiltering(plates[i],shifted, 21, 51);

                //image conversion to HSV in order to get the Saturation
                cvtColor(shifted,plateImgHSV,cv::COLOR_BGR2HSV);
               
                extractChannel(plates[i],dst,1);
                Scalar sat=cv::mean(dst);
                plate_saturation=sat[0];
                
                double min=1000;
                    double delta;
                    for(int i=0;i<filteredSamplesSaturations.size();i++){
                        delta=abs(plate_saturation-filteredSamplesSaturations[i].first);
                        if(delta<min){
                            min=delta;
                            foodFound=filteredSamplesSaturations[i].second;
                        }
                    }
                deltas.push_back(delta);

                
            }
            
            saturations_deltas_foreach_food.insert({food_name,deltas}); // ex. [pasta_with_pesto,{sat1,sat2,sat3}]...
        }

        vector<string>listFoods;

        vector<tuple<string,double,int>>min_elem_for_each_food;
       
        for(int i=0;i<plates.size();i++){
            vector<double>sat_plate_i;
            vector<string>food_dect;
                for (auto it=saturations_deltas_foreach_food.begin();it!=saturations_deltas_foreach_food.end();it++){
                    
                    food_dect.push_back(it->first);
                    sat_plate_i.push_back((it->second)[i]);//vector of saturations of the current food
                    

                }

                //find min elem index
                auto ite = std::min_element(begin(sat_plate_i), end(sat_plate_i));
                int index_food=distance(begin(sat_plate_i), ite);
                foodFound=food_dect[index_food];

            int id_food=foodClassMap[foodFound];

            if(foodFound.compare("grilled_pork")==0 || foodFound.compare("rabbit")==0 || foodFound.compare("fish_cutlet")==0 || foodFound.compare("seafood_salad")==0){
                foodFound=foodFound+"-"+side_dish_name;
            }

            listOfFoodsFounded.push_back(make_tuple(foodFound,get<0>(locations[i]),get<1>(locations[i]),id_food));
            listFoods.push_back(foodFound);
        }
        

    

        
    
        cout<<"Plate leftover founded:"<<endl;
        
        for (int w=0;w<listOfFoodsFounded.size();w++){
        
            cout<<get<0>(listOfFoodsFounded[w])<<endl;  
        }  
            
        Mat boxes=drawBoundingBoxLabel(listFoods,cloned_img);

        Mat image=boxes.clone();
        resize(image, image, Size(image.cols/3, image.rows/3)); // to half size or even smaller
       //string pathName="../CV007/ImgReport/boundingBoxDetectionImg/tray3/leftover2"; 
        //imwrite(pathName+".png",image); 
        namedWindow( "LeftOver Detection",cv::WINDOW_AUTOSIZE);
        imshow("LeftOver Detection",image);
        cout<<"Please, press a keyboard key to continue. Then wait some seconds..."<<endl;
        waitKey();
        

    }
    
   return listOfFoodsFounded;  
}


/* 
* function foodDetectionWithColor:
* function that makes the food detetion on the "before Tray"
* @param src, the image of the tray
* @return the list of the detection,a vector of :(foodName, center,radius,foodCategory)
*/
vector<tuple<string,Point,double,int>> foodDetectionWithColor(Mat src){
    cout<<"Starting the foodDetection....Please, wait some seconds\n";

    Mat cloned_img=src.clone();
    vector<tuple<Mat,Point,double>> plateAndLocation=getFoodFromPlates(src);
    vector<tuple<string,Point,double,int>> listOfFoodsFounded;
    vector<Mat> plates;
    vector<tuple<Point,double>>locations;
    string foodFound;

    for (auto const& plate : plateAndLocation){
        if(!isAGlass(plate,src)){
            plates.push_back(get<0>(plate));  
            locations.push_back(make_tuple(get<1>(plate),get<2>(plate)));
        }   
    }

    map<string,int> foodClassMap{
            {"pasta_with_pesto",1},
            {"pasta_with_tomato_sauce",2},
            {"pasta_with_meat_sauce",3},
            {"pasta_with_clams_and_mussels",4},
            {"pilaw_rice",5},
            {"grilled_pork",6},
            {"fish_cutlet",7},
            {"rabbit",8},
            {"seafood_salad",9},
            {"beans",10},
            {"basil_potatoes",11},
            {"salad",12},
            {"bread",13}
        };
   

    for(int i=0;i<plates.size();i++){
        vector<pair<double,string>> percAndPlate;
        double plate_saturation;
        Mat plateImgHSV,shifted,dst;
        foodFound="";
        
        cv::pyrMeanShiftFiltering(plates[i],shifted, 21, 51);

        //image conversion to HSV in order to get the Saturation
        cvtColor(shifted,plateImgHSV,cv::COLOR_BGR2HSV);
       
        extractChannel(plates[i],dst,1);
        Scalar sat=cv::mean(dst);
        plate_saturation=sat[0];

        map<string,double>firstCourses_map=firstCoursesDetection(shifted);

        map<string,double>secondCourses_map=secondCoursesDetection(shifted);

        map<string,double>sideDishes_map=sideDishesDetection(shifted);

        //putting all the value from the three maps into a unique one
       for (auto it=firstCourses_map.begin();it!=firstCourses_map.end();it++){
            percAndPlate.push_back({it->second,it->first});
        }
        for (auto it=secondCourses_map.begin();it!=secondCourses_map.end();it++){
             percAndPlate.push_back({it->second,it->first});
        }
        for (auto it=sideDishes_map.begin();it!=sideDishes_map.end();it++){
             percAndPlate.push_back({it->second,it->first});
        }

       
        sort(percAndPlate.rbegin(), percAndPlate.rend());
        

        //take the 6 bigger percentuals
        int numElem=6;

        vector<pair<double,string>> higherPerc;
        for(int i=0;i<numElem;i++){
            higherPerc.push_back({percAndPlate[i].first,percAndPlate[i].second});
        }

        //Let's take the saturation values for the sample of the food 
        //with the higher percentual
        map<string,double> sampleSaturations=getSamplesSaturation();

    
        vector<pair<double,string>> splSatForHigherPerc;//sampleSaturationForHigherPerc
        for(int i=0;i<numElem;i++){
            string food_name=higherPerc[i].second;
            if(food_name.find("salad")!=string::npos){
                if(food_name.find("seafood_salad")!=string::npos){
                    splSatForHigherPerc.push_back({sampleSaturations[food_name],food_name});
                }
                else{
                    splSatForHigherPerc.push_back({sampleSaturations[food_name+"1"],food_name});
                    splSatForHigherPerc.push_back({sampleSaturations[food_name+"2"],food_name});
                    splSatForHigherPerc.push_back({sampleSaturations[food_name+"3"],food_name});
                    splSatForHigherPerc.push_back({sampleSaturations[food_name+"4"],food_name});
                }
               
            }
            else{
                if(food_name.find("fish_cutlet")!=string::npos || food_name.find("pasta_with_clams_and_mussels")!=string::npos ){
                    splSatForHigherPerc.push_back({sampleSaturations[food_name+"1"],food_name});
                    splSatForHigherPerc.push_back({sampleSaturations[food_name+"2"],food_name});
                    splSatForHigherPerc.push_back({sampleSaturations[food_name+"3"],food_name});
                }
                else{
                    if(food_name.find("grilled_pork")!=string::npos || food_name.find("rabbit")!=string::npos || food_name.find("pasta_with_tomato_sauce")!=string::npos){
                        splSatForHigherPerc.push_back({sampleSaturations[food_name+"1"],food_name});
                        splSatForHigherPerc.push_back({sampleSaturations[food_name+"2"],food_name});
                    }
                    else{
                         splSatForHigherPerc.push_back({sampleSaturations[food_name],food_name});
                    }
                }
               
            }
        }

       

        double min=1000;
        for(int i=0;i<splSatForHigherPerc.size();i++){
            double delta=abs(plate_saturation-splSatForHigherPerc[i].first);
            if(delta<min){
                min=delta;
                foodFound=splSatForHigherPerc[i].second;
            }
        }

      
       string side_dish="";
       int id_food=foodClassMap[foodFound];
       if(foodFound.compare("grilled_pork")==0 || foodFound.compare("rabbit")==0 || foodFound.compare("fish_cutlet")==0 || foodFound.compare("seafood_salad")==0){
            side_dish=getSideDish(shifted);
            foodFound=foodFound+"-"+side_dish;
       }

        listOfFoodsFounded.push_back(make_tuple(foodFound,get<0>(locations[i]),get<1>(locations[i]),id_food));
    }
   
    vector<string>listFoods;
    for (int i=0;i<listOfFoodsFounded.size();i++){
        cout<<get<0>(listOfFoodsFounded[i])<<endl;  
        listFoods.push_back(get<0>(listOfFoodsFounded[i]));
    }  

    Mat boxes=drawBoundingBoxLabel(listFoods,cloned_img);
    Mat image=boxes.clone();
        resize(image, image, Size(image.cols/3, image.rows/3)); // to half size or even smaller
        /*string pathName="../CV007/ImgReport/boundingBoxDetectionImg/tray3/food_image"; 
        imwrite(pathName+".png",image); */
        namedWindow( "food Detection",cv::WINDOW_AUTOSIZE);
    imshow("food Detection",image);
    cout<<"Please, press a keyboard key to continue. Then wait some seconds..."<<endl;
    waitKey();
        
    return listOfFoodsFounded;
}



/*
* function getSideDish
* function that is used to find the presence of a possible side dish into a secondCourse plate
* @param secondCourseImg, image of the second course plate
* @return the name of the side dish found
*/
string getSideDish(Mat secondCourseImg){

    Mat srcHSV;
    cvtColor(secondCourseImg,srcHSV,cv::COLOR_BGR2HSV);

    double dimImg=secondCourseImg.cols*secondCourseImg.rows;
    map<string,double>food_percCoverImg;
    struct foodColor fc;
    double numColoredPixel=0;
    Mat mask;
    double totPerc=0;
    double totColoredPixel=0;
  
    
     //beans----------------------------------
    double percOfCoverImg=0;
    inRange(srcHSV,fc.peal_color_min,fc.peal_color_max,mask);
    numColoredPixel=countNonZero(mask);
    percOfCoverImg=(numColoredPixel*100)/dimImg;
    totPerc+=percOfCoverImg;
    food_percCoverImg.insert({"beans",numColoredPixel});  
   

    //potatoes---------------------------------------
   
    numColoredPixel=0;
    inRange(srcHSV,fc.yellow_potatoes_min,fc.yellow_potatoes_max,mask);
    numColoredPixel=countNonZero(mask);
    percOfCoverImg=(numColoredPixel*100)/dimImg;
    totPerc+=percOfCoverImg;
    food_percCoverImg.insert({"basil_potatoes",numColoredPixel});   
    

    string nameOffoodFound="";
    if(totPerc>=10.0){ //sideDish is present if totPerc>=10%
        double max=0;
   
        for (auto it=food_percCoverImg.begin();it!=food_percCoverImg.end();it++){
            if((it->second)>=max){
                max=it->second;
                nameOffoodFound=it->first;
            }
        }
    }

    return nameOffoodFound;
}


/*
* function getSamplesSaturation:
* fucntion that for each sample contained into the folder DATASET (see above), compute the saturation 
* @return a map containing for each sample, the relative saturation value
*/
map<string,double> getSamplesSaturation(){
    map<string,Mat> samples_imgs=readDataset();

    map<string,double> samples_imgs_saturations;
    for (auto it=samples_imgs.begin();it!=samples_imgs.end();it++){

        //Take the name of the food from the path
        string name=it->first;
        string s=".jpg";
        int index=name.find(s);
        if (index != std::string::npos)
            name.erase(index, s.length());
        name=name.substr(DATASET.length()+1);
    
        Mat img=it->second;
        Mat imgHSV,shifted,dst;
        

        extractChannel(img,dst,1);
        Scalar sat=cv::mean(dst);
       
        samples_imgs_saturations.insert({name,sat[0]});
        
    }
    return samples_imgs_saturations;
}

/*
* function firstCoursesDetection:
* function that return the number of pixel found for each first courses color, 
* the number of pixel founded into the range
* @param src, the tray into which make the detection
* @return  for each main course colors range, the relative number of pixel found into the tray
*/
map<string,double> firstCoursesDetection(Mat src){
    Mat srcHSV;
    cvtColor(src,srcHSV,cv::COLOR_BGR2HSV);

    double dimImg=src.cols*src.rows;
    map<string,double>food_percCoverImg;
    struct foodColor fc;
    double numColoredPixel;
    Mat mask;
    
    double totPerc;

    //pilaw_rice------------------------------------
    double percOfCoverImg=0;//% of colored image 
    numColoredPixel=0;
    inRange(srcHSV,fc.yellow_rice_min,fc.yellow_rice_max,mask);
    numColoredPixel+=countNonZero(mask);

    inRange(srcHSV,fc.green_peas_min,fc.green_peas_max,mask);
    numColoredPixel+=countNonZero(mask);
    

    inRange(srcHSV,fc.yellow_pepper_min,fc.yellow_pepper_max,mask);
    numColoredPixel+=countNonZero(mask);
  

    inRange(srcHSV,fc.red_pepper_min,fc.red_pepper_max,mask);
    numColoredPixel+=countNonZero(mask);
  
    
    food_percCoverImg.insert({"pilaw_rice",numColoredPixel});
    //------------------------------------------------------

    //pasta_with_pesto--------------------------------------
    percOfCoverImg=0;
    numColoredPixel=0;
    inRange(srcHSV,fc.yellow_light_min,fc.yellow_light_max,mask);
    numColoredPixel+=countNonZero(mask);
    percOfCoverImg+=(numColoredPixel*100)/dimImg;
   
    inRange(srcHSV,fc.green_pesto_min,fc.green_pesto_max,mask);
    numColoredPixel+=countNonZero(mask);
   

    food_percCoverImg.insert({"pasta_with_pesto",numColoredPixel});
    //------------------------------------------------------

    //pasta_with_tomato_sauce------------------------------
    percOfCoverImg=0;
    numColoredPixel=0;
    inRange(srcHSV,fc.yellow_light_min,fc.yellow_light_max,mask);
    numColoredPixel+=countNonZero(mask);

    inRange(srcHSV,fc.red_sauce_min,fc.red_sauce_max,mask);
    numColoredPixel+=countNonZero(mask);
   

    food_percCoverImg.insert({"pasta_with_tomato_sauce",numColoredPixel});
    //------------------------------------------------------


    //pasta_with_meat_sauce
    percOfCoverImg=0;
    numColoredPixel=0;
    inRange(srcHSV,fc.yellow_light_min,fc.yellow_light_max,mask);
    numColoredPixel+=countNonZero(mask);

    inRange(srcHSV,fc.meat_sauce_min,fc.meat_sauce_max,mask);
    numColoredPixel=countNonZero(mask);

    food_percCoverImg.insert({"pasta_with_meat_sauce",numColoredPixel});
    //------------------------------------------------------

    //pasta_with_clams_and_mussels
    percOfCoverImg=0;
    numColoredPixel=0;
    inRange(srcHSV,fc.orange_min,fc.orange_max,mask);
    numColoredPixel+=countNonZero(mask);
  
    food_percCoverImg.insert({"pasta_with_clams_and_mussels",numColoredPixel});        
    //------------------------------------------------------
    
    return food_percCoverImg;
}


/*
* function secondCoursesDetection:
* function that return the number of pixel found for each second courses color, 
* the number of pixel founded into the range
* @param src, the tray into which make the detection
* @return  for each main course colors range, the relative number of pixel found into the tray
*/

map<string,double> secondCoursesDetection(Mat src){
    Mat srcHSV;
    cvtColor(src,srcHSV,cv::COLOR_BGR2HSV);

    double dimImg=src.cols*src.rows;
    map<string,double>food_percCoverImg;
    struct foodColor fc;
    double numColoredPixel=0;
    Mat mask;
    
    double percOfCoverImg=0;//% of colored image 
   
    //fish_cutlet and potatoes---------------------------
    percOfCoverImg=0;
    numColoredPixel=0;
    inRange(srcHSV,fc.orange_light_min,fc.orange_light_max,mask);
    numColoredPixel+=countNonZero(mask);
    inRange(srcHSV,fc.yellow_potatoes_min,fc.yellow_potatoes_max,mask);
    numColoredPixel+=countNonZero(mask);
    food_percCoverImg.insert({"fish_cutlet",numColoredPixel});  

    //rabbit and beans------------------------------------
    percOfCoverImg=0;
    numColoredPixel=0;
    inRange(srcHSV,fc.brown_min,fc.brown_max,mask);
    numColoredPixel+=countNonZero(mask);
    inRange(srcHSV,fc.peal_color_min,fc.peal_color_max,mask);
    numColoredPixel+=countNonZero(mask);
    food_percCoverImg.insert({"rabbit",numColoredPixel});   

    
    //fish_salad, beans and potatoes--------------------------------------
    percOfCoverImg=0;
    numColoredPixel=0;
    inRange(srcHSV,fc.pink_piece_min,fc.pink_piece_max,mask);
    numColoredPixel+=countNonZero(mask);    
    inRange(srcHSV,fc.purple_piece_min,fc.purple_piece_max,mask);
    numColoredPixel+=countNonZero(mask);
    inRange(srcHSV,fc.white_piece_min,fc.white_piece_max,mask);
    numColoredPixel+=countNonZero(mask);
    inRange(srcHSV,fc.yellow_potatoes_min,fc.yellow_potatoes_max,mask);
    numColoredPixel+=countNonZero(mask);
    inRange(srcHSV,fc.peal_color_min,fc.peal_color_max,mask);
    numColoredPixel+=countNonZero(mask);
    food_percCoverImg.insert({"seafood_salad",numColoredPixel});

    //grilled_pork and beans------------------------------------------
    percOfCoverImg=0;
    numColoredPixel=0;
    inRange(srcHSV,fc.white_meat_min,fc.white_meat_max,mask);
    numColoredPixel+=countNonZero(mask);
    inRange(srcHSV,fc.peal_color_min,fc.peal_color_max,mask);
    numColoredPixel+=countNonZero(mask);
    inRange(srcHSV,fc.mushrooms_min,fc.mushrooms_max,mask);
    numColoredPixel+=countNonZero(mask);
    food_percCoverImg.insert({"grilled_pork",numColoredPixel});  

     //bread---------------------------------------
    percOfCoverImg=0;
    numColoredPixel=0;
    inRange(srcHSV,fc.brown_light_min,fc.brown_light_max,mask);
    numColoredPixel+=countNonZero(mask);
    food_percCoverImg.insert({"bread",numColoredPixel});  

    return food_percCoverImg;
    
}

/*
* function sideDishesDetection:
* function that return the number of pixel found for each side dishes (in the function there is only the salad) color, 
* the number of pixel founded into the range.
* Here the sideDishes are considered all the dishes that are in a separate plate that they are not first courses or second courses
* @param src, the tray into which make the detection
* @return  for each main course colors range, the relative number of pixel found into the tray
*/

map<string,double> sideDishesDetection(Mat src){
    Mat srcHSV;
    cvtColor(src,srcHSV,cv::COLOR_BGR2HSV);

    double dimImg=src.cols*src.rows;
    map<string,double>food_percCoverImg;
    struct foodColor fc;
    double numColoredPixel=0;
    Mat mask;
    
    double percOfCoverImg=0;//% of colored image 

         
    //side dishes
   
    //salad---------------------------
    percOfCoverImg=0;
    numColoredPixel=0;
    inRange(srcHSV,fc.red_tomato_min,fc.red_tomato_max,mask);
    numColoredPixel+=countNonZero(mask);

    inRange(srcHSV,fc.green_salad_min,fc.green_salad_max,mask);
    numColoredPixel+=countNonZero(mask);

    inRange(srcHSV,fc.purple_salad_min,fc.purple_salad_max,mask);
    numColoredPixel+=countNonZero(mask);

    inRange(srcHSV,fc.orange_carrot_min,fc.orange_carrot_max,mask);
    numColoredPixel+=countNonZero(mask);

    inRange(srcHSV,fc.yellow_pepper_min,fc.yellow_pepper_max,mask);
    numColoredPixel+=countNonZero(mask);

    food_percCoverImg.insert({"salad",numColoredPixel}); 

 

    return food_percCoverImg;
    
}


/*
* function getFoodFromPlates:
* function that allow us to obtain a tary where only the foods are visible. hence it erase what in the background is not food.
* Then it separate each plate in a single image and put it into a vector
* @param img, tray image
* @return a vector containing the image of the single plate, it's center and radius
*/

vector<tuple<Mat,Point,double>> getFoodFromPlates(Mat img){
    cv::Mat gray;
   

  
   vector<tuple<Mat,Point,double>> platesAndLocation;
   const int WHITE_PLATE_RADIUS=280;
   cvtColor(img, gray, cv::COLOR_BGR2GRAY );


   //apply a gaussian filter for the HoughCircles function
   cv::GaussianBlur( gray, gray, cv::Size(7,7), 0 );

   //vector that stores alle the circles identified by HoughCircles
   std::vector<cv::Vec3f> circles;
   HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, 420, 200, 50, 0, 0);



   vector<Point> centers;
   vector<int> radiuses;
   for( size_t i = 0; i < circles.size(); i++ ){
      cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      centers.push_back(center);
      radiuses.push_back(radius);
     
   }


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
                img.at<cv::Vec3b>(i,j)[0] = 200;
                img.at<cv::Vec3b>(i,j)[1] = 200;
                img.at<cv::Vec3b>(i,j)[2] = 200;
            } else{
                    double valA = static_cast<double>(img.at<cv::Vec3b>(i,j)[0]);
                    double valB = static_cast<double>(img.at<cv::Vec3b>(i,j)[1]);
                    double valC = static_cast<double>(img.at<cv::Vec3b>(i,j)[2]);
                    double d1 = abs((static_cast<double>(valA/valB))-1);
                    double d2 = abs((static_cast<double>(valC/valB))-1);
                    double d3 = abs((static_cast<double>(valA/valC))-1);

                    if(d1 <= Threshold && d2 <= Threshold && d3 <= Threshold){
                        img.at<cv::Vec3b>(i,j)[0] = 200;
                        img.at<cv::Vec3b>(i,j)[1] = 200;
                        img.at<cv::Vec3b>(i,j)[2] = 200;
                    }
                }
      }
   }


   

    vector<tuple<Rect, Mat>> images;
	vector<Mat>foodPlates;

    for(int i=0;i<circles.size();i++){
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        //create a new image that has dimensione diameter*diameter
        Mat temp(radius*2,radius*2,CV_8UC3,Scalar(0,0,0));

        vector<cv::Point> contour;
        contour.push_back(Point((center.x)-radius, (center.y)-radius));
        contour.push_back(Point((center.x)+radius, (center.y)-radius));
        contour.push_back(Point((center.x)+radius, (center.y)+radius));
        contour.push_back(Point((center.x)-radius, (center.y)+radius));

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
        
      
		Mat crop_img = temp;
		tuple <Rect, Mat> t;
        t = make_tuple(box, crop_img);
        images.push_back(t);
       
    }

    
    Mat mask2 = Mat::zeros(img.size(), CV_8UC1);
    Mat fgModel = Mat::zeros(1, 65, CV_64FC1);
    Mat bgModel = Mat::zeros(1, 65, CV_64FC1);

    Mat dst1;

    for (tuple<Rect, Mat> t : images){
        int x, y, w, h;
        x = get<0>(t).x;
        y = get<0>(t).y;
        w = get<0>(t).width;
        h = get<0>(t).height;

        grabCut(img, mask2, Rect(x, y, w, h), bgModel, fgModel, 1, GC_INIT_WITH_RECT);
        Mat mask3 = (mask2 == 3);

        img.copyTo(dst1, mask3);
    }

    for(int i=0;i<circles.size();i++){
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        if(radius>=WHITE_PLATE_RADIUS){
            radius= (cvRound(circles[i][2])*4)/5;//let's take the 4/5 of the radius in order to take only the food and not the plate contour
        }
    
        Mat temp(radius*2,radius*2,CV_8UC3,Scalar(0,0,0));

        int x=0;
        for(int i=(center.y)-radius;(i<=(center.y)+radius)&&(x<temp.rows);i++){
            int y=0;
            for(int j=(center.x)-radius;(j<=(center.x)+radius)&&(y<temp.cols);j++){
                temp.at<Vec3b>(y,x)[0]=dst1.at<Vec3b>(i,j)[0];
                temp.at<Vec3b>(y,x)[1]=dst1.at<Vec3b>(i,j)[1];
                temp.at<Vec3b>(y,x)[2]=dst1.at<Vec3b>(i,j)[2];
                y++;
            }
            x++;
        }
        
        platesAndLocation.push_back(make_tuple(temp, center, radius));
    }

    return platesAndLocation;
   
}







/******************************************************************/
/********************UTILITY FUNCTIONS*****************************/
/******************************************************************/

/*function readDataset
*function used to read data from DATASET (see above)
*@return the map of images read on the Dataset
*/

map<string,Mat> readDataset(){
    map<string,Mat> samples_imgs;
    vector<string> img_names;
    cv::utils::fs::glob(DATASET,FORMAT,img_names);
    for(int i=0;i<img_names.size();i++){
        samples_imgs.insert({img_names[i],imread(img_names[i])});
    }
    return samples_imgs;

}

/*function houghTrasform
*function that use the houghTrasform in order to find the plates into the input image
*@param img, img where to find the circles
*@return img, qhere the circle are drawn
*/
Mat houghTrasform(Mat img){


   //get the gray scale of the original image
   cv::Mat gray;
   cvtColor( img, gray, cv::COLOR_BGR2GRAY );
   cv::GaussianBlur( gray, gray, cv::Size(7,7), 0 );
   std::vector<cv::Vec3f> circles;
   HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, 420, 200, 50, 0, 0);

   vector<Point> centers;
   vector<int> radiuses;
   // Draw the circles
   for( size_t i = 0; i < circles.size(); i++ ){
      cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      centers.push_back(center);
      radiuses.push_back(radius);
      circle( img, center, 2, cv::Scalar(0,0,0), -1, 8, 0 );
      circle( img, center, radius, cv::Scalar(0,0,0), 2, 0, 0 );
   }

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
            img.at<cv::Vec3b>(i,j)[0] = 0;
            img.at<cv::Vec3b>(i,j)[1] = 0;
            img.at<cv::Vec3b>(i,j)[2] = 0;
         }
      }
   }
    return img;
}

/*
* function extractRangeCOlorFromPlate
* function that allow to find the HVS range of a certain element into the imput image, by 
* working with a set of trackbar
* @param src, source image
* @return the image where the range found is applied
*/

Mat extractRangeCOlorFromPlate(Mat src){
    auto const MASK_WINDOW="Mask_settings";
    cv::namedWindow(MASK_WINDOW,WINDOW_AUTOSIZE);

    //HSV range default range
    int minH=0; int maxH=120;
    int minSat=75; int maxSat=255;
    int minVal=0; int maxVal=255;
   

    createTrackbar("Min Hue",MASK_WINDOW,&minH,179);
    createTrackbar("Min Sat",MASK_WINDOW,&minSat,255);
    createTrackbar("Min Val",MASK_WINDOW,&minVal,255);
    createTrackbar("Max Hue",MASK_WINDOW,&maxH,179);
    createTrackbar("Max Sat",MASK_WINDOW,&maxSat,255);
    createTrackbar("Max Val",MASK_WINDOW,&maxVal,255);
    
    
    Mat result;
    while(true){
        Mat srcHSV,mask,resultImage;
        cvtColor(src,srcHSV,cv::COLOR_BGR2HSV);
       
        inRange(srcHSV,Scalar(minH,minSat,minVal),Scalar(maxH,maxSat,maxVal),mask);
        bitwise_and(src,src,resultImage,mask);

        imshow("Input image: ",src);
        imshow("Result (masked) image: ",resultImage);
        
        //press "enter" to break the loop
        if(waitKey(30)==27){
            result=resultImage;
            cout<<"min["<<minH<<","<<minSat<<","<<minVal<<"] ";
            cout<<"max["<<maxH<<","<<maxSat<<","<<maxVal<<"]"<<endl;
            break;
        }
    }
    return result; 
}






