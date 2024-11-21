
/*
* @author Carlotta Schiavo,ID=2076743 and Qiu yi jian, 	ID=2085730
*/

#include "metricsLibs.h"

using namespace cv;
using namespace std;
using namespace  std::experimental::filesystem;

double foodLeftOverMetrics(Mat before, Mat after){

   Mat mask = Mat::zeros(before.size(), CV_8U);
   double numPixelBefore = 0;
   double numPixelAfter = 0;

   //compute numer of pixels in before image
   inRange(before, 1, 255, mask);
   numPixelBefore += countNonZero(mask);

   //compute numer of pixels in after image
   mask = Mat::zeros(before.size(), CV_8U);
   inRange(after, 1, 255, mask);
   numPixelAfter += countNonZero(mask);

   return numPixelAfter/numPixelBefore;
}

/*mIoU

*/

double mIoU(Mat prediction,Mat gt){

    std::map<std::string,int> foodClassMap{
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

    cvtColor(prediction,prediction,cv::COLOR_BGR2GRAY);
    cvtColor(gt,gt,cv::COLOR_BGR2GRAY);

    map<int,Mat>mask_predictions;
    map<int,Mat>mask_gt;
    vector<int>ID_gt;

    int i=1;
    for (auto it=foodClassMap.begin();it!=foodClassMap.end();it++){
        Mat mask_p = Mat::zeros(prediction.size(), CV_8U);
        Mat mask_g= Mat::zeros(gt.size(), CV_8U);
        
        inRange(prediction, i, i, mask_p);
        inRange(gt, i, i, mask_g);
        mask_predictions.insert({i,mask_p});
        mask_gt.insert({i,mask_g});

        if( countNonZero(mask_g)>0){
            ID_gt.push_back(i);
        }
        i++;
    }

    //calculation of insersection for each class founded into the GT
    vector<double>intersectionBetweenClasses;
    vector<double>unionBetweenClasses;

    for(auto it=mask_gt.begin();it!=mask_gt.end();it++){
        int id_class=it->first;
        Mat mask_p = mask_predictions[id_class];
        Mat mask_g= it->second;
        Mat mask_intersection=Mat::zeros(prediction.size(), CV_8U);
        Mat mask_union=Mat::zeros(prediction.size(), CV_8U);

        

        bitwise_and(mask_g,mask_p,mask_intersection);
        bitwise_or(mask_g,mask_p,mask_union);

        double numPixelAfter = countNonZero(mask_intersection);
        intersectionBetweenClasses.push_back(numPixelAfter);
        numPixelAfter=countNonZero(mask_union);
        unionBetweenClasses.push_back(numPixelAfter);
    }

    double sumIoU;


    for(int id:ID_gt){
        sumIoU+=intersectionBetweenClasses[id-1]/unionBetweenClasses[id-1];
    }

    double mIoU=sumIoU/ID_gt.size();



   return mIoU;
}







double mAP(){
    double mAP=0;
    vector<map<int,Mat>> gt_boxes;

    vector<map<int,Mat>> predicted_boxes;

    cout<<"Computation of GT boxes"<<endl;
    gt_boxes=readGTboxes();
    cout<<"Reading the images of the tray"<<endl;
    vector<Mat> traysImages=getTraysImages();
    cout<<"Reading the images of the tray"<<endl;
    Mat src=imread("../GroundTruth/tray3/food_image.jpg");

    vector<vector<string>>predected_foods_lists{
       //Tray1
        /*{"pasta_with_pesto","grilled_pork-beans"},
        {"grilled_pork-beans","pasta_with_pesto","grilled_pork-beans"},
        {"pasta_with_pesto","grilled_pork-beans"},*/
        //Tray2
       /* {"pasta_with_tomato_sauce","salad","fish_cutlet-beans"},
        {"pasta_with_tomato_sauce","fish_cutlet-beans","salad"},
        {"fish_cutlet-beans","salad","salad"},

        //Tray3
        {"pasta_with_tomato_sauce","pasta_with_pesto","rabbit-"},
        {"pasta_with_tomato_sauce","rabbit-","pasta_with_pesto"},
        {"rabbit-","pasta_with_tomato_sauce","pasta_with_tomato_sauce"},

        //Tray4
        {"fish_cutlet-basil_potatoes","salad","pasta_with_pesto"},
        {"fish_cutlet-basil_potatoes","salad","fish_cutlet-basil_potatoes"},
        {"pasta_with_pesto","fish_cutlet-basil_potatoes","salad"},

        //Tray5
        { "fish_cutlet-basil_potatoes", "rabbit-beans"},
        {"fish_cutlet-basil_potatoes", "rabbit-beans"},
        {"fish_cutlet-basil_potatoes"},

        //Tray6
        {"grilled_pork-beans","salad","pasta_with_clams_and_mussels"},
        {"pasta_with_clams_and_mussels","grilled_pork-beans","salad","pasta_with_clams_and_mussels"},
        {"pasta_with_clams_and_mussels","pasta_with_clams_and_mussels","grilled_pork-beans","pasta_with_clams_and_mussels"},
        
        //Tray7
        {"fish_cutlet-basil_potatoes", "pasta_with_clams_and_mussels", "pasta_with_tomato_sauce"},
        {"pasta_with_clams_and_mussels","fish_cutlet-basil_potatoes","pasta_with_clams_and_mussels","pasta_with_tomato_sauce"},
        {"fish_cutlet-", "fish_cutlet-", "fish_cutlet-", "fish_cutlet-"},

        //Tray8
        {"pasta_with_clams_and_mussels", "seafood_salad-beans", "salad", "rabbit-"},
        {"seafood_salad-beans", "pasta_with_clams_and_mussels", "rabbit-", "salad"},
        {"salad", "pasta_with_clams_and_mussels", "pasta_with_clams_and_mussels"}*/



         //Tray1
       /* {"pasta_with_pesto","grilled_pork","beans"},
        {"grilled_pork","pasta_with_pesto","grilled_pork","beans"},
        {"pasta_with_pesto","grilled_pork","beans"},
        //Tray2
        {"pasta_with_tomato_sauce","salad","fish_cutlet","beans"},
        {"pasta_with_tomato_sauce","fish_cutlet","beans","salad"},
        {"fish_cutlet","beans","salad","salad"},

        //Tray3
        {"pasta_with_tomato_sauce","pasta_with_pesto","rabbit"},
        {"pasta_with_tomato_sauce","rabbit","pasta_with_pesto"},
        {"rabbit","pasta_with_tomato_sauce","pasta_with_tomato_sauce"},

        //Tray4
        {"fish_cutlet-basil_potatoes","salad","pasta_with_pesto"},
        {"fish_cutlet-basil_potatoes","salad","fish_cutlet","basil_potatoes"},
        {"pasta_with_pesto","fish_cutlet","basil_potatoes","salad"},

        //Tray5
        { "fish_cutlet","basil_potatoes", "rabbit","beans"},
        {"fish_cutlet","basil_potatoes", "rabbit","beans"},
        {"fish_cutlet","basil_potatoes"},

        //Tray6
        {"grilled_pork","beans","salad","pasta_with_clams_and_mussels"},
        {"pasta_with_clams_and_mussels","grilled_pork","beans","salad","pasta_with_clams_and_mussels"},
        {"pasta_with_clams_and_mussels","pasta_with_clams_and_mussels","grilled_pork","beans","pasta_with_clams_and_mussels"},
        
        //Tray7
        {"fish_cutlet-basil_potatoes", "pasta_with_clams_and_mussels", "pasta_with_tomato_sauce"},
        {"pasta_with_clams_and_mussels","fish_cutlet","basil_potatoes","pasta_with_clams_and_mussels","pasta_with_tomato_sauce"},
        {"fish_cutlet", "fish_cutlet", "fish_cutlet", "fish_cutlet"},

        //Tray8
        {"pasta_with_clams_and_mussels", "seafood_salad","beans", "salad", "rabbit"},
        {"seafood_salad","beans", "pasta_with_clams_and_mussels", "rabbit", "salad"},
        {"salad", "pasta_with_clams_and_mussels", "pasta_with_clams_and_mussels"}*/

        //Tray1
        {"pasta_with_pesto","grilled_pork"},
        {"grilled_pork","pasta_with_pesto","grilled_pork"},
        {"pasta_with_pesto","grilled_pork"},
        //Tray2
        {"pasta_with_tomato_sauce","salad","fish_cutlet",},
        {"pasta_with_tomato_sauce","fish_cutlet","salad"},
        {"fish_cutlet","salad","salad"},

        //Tray3
        {"pasta_with_tomato_sauce","pasta_with_pesto","rabbit"},
        {"pasta_with_tomato_sauce","rabbit","pasta_with_pesto"},
        {"rabbit","pasta_with_tomato_sauce","pasta_with_tomato_sauce"},

        //Tray4
        {"fish_cutlet","salad","pasta_with_pesto"},
        {"fish_cutlet","salad","fish_cutlet"},
        {"pasta_with_pesto","fish_cutlet","salad"},

        //Tray5
        { "fish_cutlet", "rabbit"},
        {"fish_cutlet", "rabbit"},
        {"fish_cutlet"},

        //Tray6
        {"grilled_pork","salad","pasta_with_clams_and_mussels"},
        {"pasta_with_clams_and_mussels","grilled_pork","salad","pasta_with_clams_and_mussels"},
        {"pasta_with_clams_and_mussels","pasta_with_clams_and_mussels","grilled_pork","pasta_with_clams_and_mussels"},
        
        //Tray7
        {"fish_cutlet-basil_potatoes", "pasta_with_clams_and_mussels", "pasta_with_tomato_sauce"},
        {"pasta_with_clams_and_mussels","fish_cutlet","pasta_with_clams_and_mussels","pasta_with_tomato_sauce"},
        {"fish_cutlet", "fish_cutlet", "fish_cutlet", "fish_cutlet"},

        //Tray8
        {"pasta_with_clams_and_mussels", "seafood_salad", "salad", "rabbit"},
        {"seafood_salad", "pasta_with_clams_and_mussels", "rabbit", "salad"},
        {"salad", "pasta_with_clams_and_mussels", "pasta_with_clams_and_mussels"}

    };


  
    for(int i=0;i<predected_foods_lists.size();i++){
        cout<<"Computation of predicted boxes\n";
        map<int,Mat> t=drawBoundingBoxLabel_For_metrics(predected_foods_lists[i],traysImages[i]);
        predicted_boxes.push_back(t);
    }

  
    
    map<int,vector<pair<double,double>>>recAndPrec{
                {1,{}},
                {2,{}},
                {3,{}},
                {4,{}},
                {5,{}},
                {6,{}},
                {7,{}},
                {8,{}},
                {9,{}},
                {10,{}},
                {11,{}},
                {12,{}},
                {13,{}}
            };
    for(int i=0;i<gt_boxes.size();i++){
        for(auto it=gt_boxes[i].begin();it!=gt_boxes[i].end();it++){
            int id_gt=it->first;
            Mat temp=Mat::zeros(src.size(), CV_8U);
            if(!predicted_boxes[i].count(id_gt)>0){
                predicted_boxes[i].insert({id_gt,temp});
            }
        }
    }

    for(int i=0;i<predicted_boxes.size();i++){
        for(auto it=predicted_boxes[i].begin();it!=predicted_boxes[i].end();it++){
            int id=it->first;
            Mat temp=Mat::zeros(src.size(), CV_8U);
            if(!gt_boxes[i].count(id)>0){
                gt_boxes[i].insert({id,temp});
            }
           
        }
    }

    cout<<"boxes_gt:"<<endl;
    for(int i=0;i<gt_boxes.size();i++){
        cout<<"Keys:"<<endl;
        for(auto it=gt_boxes[i].begin();it!=gt_boxes[i].end();it++){
           cout<<it->first<<" "<<"size: "<<it->second.size()<<endl;

        }
    }

    cout<<"predicted_boxes:"<<endl;
    for(int i=0;i<predicted_boxes.size();i++){
        cout<<"Keys:"<<endl;
        for(auto it=predicted_boxes[i].begin();it!=predicted_boxes[i].end();it++){
           cout<<it->first<<" "<<"size: "<<it->second.size()<<endl;

        }
    }


    
    map<int,vector<double>>intersectionBetweenClasses{
        {1,{}},
        {2,{}},
        {3,{}},
        {4,{}},
        {5,{}},
        {6,{}},
        {7,{}},
        {8,{}},
        {9,{}},
        {10,{}},
        {11,{}},
        {12,{}},
        {13,{}}
    };
    map<int,vector<double>>unionBetweenClasses{
        {1,{}},
        {2,{}},
        {3,{}},
        {4,{}},
        {5,{}},
        {6,{}},
        {7,{}},
        {8,{}},
        {9,{}},
        {10,{}},
        {11,{}},
        {12,{}},
        {13,{}}
    };

          

        for(int i=0;i<gt_boxes.size();i++){
            cout<<"Fuori"<<endl;
            for(auto it=gt_boxes[i].begin();it!=gt_boxes[i].end();it++){
                   cout<<"Computation of intersections and unions for each class"<<endl;

                    int id_class=it->first;
                    Mat mask_g= it->second;
                    Mat mask_p = predicted_boxes[i][id_class];

                   
                  
                    Mat mask_pb;
                    Mat mask_gb;


                    Mat mask_intersection=Mat::zeros(mask_p.size(), CV_8UC1);
                    Mat mask_union=Mat::zeros(mask_p.size(), CV_8UC1);

                   


                    bitwise_and(mask_g,mask_p,mask_intersection);
                    bitwise_or(mask_g,mask_p,mask_union);
                

                    double numPixelAfter = countNonZero(mask_intersection);
                    cout<<"numPixIntersection "<<numPixelAfter<<endl;

                    intersectionBetweenClasses[id_class].push_back(numPixelAfter);
                    numPixelAfter=countNonZero(mask_union);
                    cout<<"numPixUnion "<<numPixelAfter<<endl;
                    unionBetweenClasses[id_class].push_back(numPixelAfter);
            }
            
        }
       
            double threshold=0.5;
           
            map<int,vector<double>>IoU{
                {1,{}},
                {2,{}},
                {3,{}},
                {4,{}},
                {5,{}},
                {6,{}},
                {7,{}},
                {8,{}},
                {9,{}},
                {10,{}},
                {11,{}},
                {12,{}},
                {13,{}}
            };
            
                for(int id=1;id<14;id++){
                    for(int i=0;i<intersectionBetweenClasses[id].size();i++){

                        IoU[id].push_back((intersectionBetweenClasses[id][i]/unionBetweenClasses[id][i]));
                    }
                }


                cout<<"IoU map:"<<endl;
                for(auto it=IoU.begin();it!=IoU.end();it++){
                    cout<<it->first<<"-";
                    for(int i=0;i<it->second.size();i++){
                         cout<<it->second[i]<<endl;;
                    }
                }


                 for(int id=1;id<14;id++){              
                    for(int i=0;i<IoU[id].size();i++){
                        
                        int confusionMatrix[]={0,0,0};// i=0=>TP, i=1=>FP, i=2=>FN
                            if(IoU[id][i]>threshold){
                                confusionMatrix[0]++;
                            }
                            else if(IoU[id][i]<=threshold && IoU[id][i]>0){
                                confusionMatrix[1]++;
                            }
                            else{
                                confusionMatrix[2]++;
                            }
                        double Recall=confusionMatrix[0]/(confusionMatrix[0]+confusionMatrix[2]);
                        double Precision=confusionMatrix[0]/(confusionMatrix[0]+confusionMatrix[1]);

                        recAndPrec[id].push_back({Recall,Precision});

                    }
                }


        cout<<"recAndPrec:"<<endl;
        for(int i=0;i<recAndPrec.size();i++){
            cout<<"Keys "<<i<<endl;
            for(auto it=recAndPrec[i].begin();it!=recAndPrec[i].end();it++){
                cout<<(it->first)<<" "<<(it->second)<<endl;
            }
        }
                    

            
        //order in ascending 

        map<int,vector<pair<double,double>>>recAndPrec_ordered{
                {1,{}},
                {2,{}},
                {3,{}},
                {4,{}},
                {5,{}},
                {6,{}},
                {7,{}},
                {8,{}},
                {9,{}},
                {10,{}},
                {11,{}},
                {12,{}},
                {13,{}}
            };

        for(auto it=recAndPrec.begin();it!=recAndPrec.end();it++){
            int id=it->first;
            vector<pair<double,double>> temp=it->second;

            sort(temp.begin(),temp.end(),comparePairs);
            recAndPrec_ordered.insert({id,temp});
            
        }

        map<int,vector<double>>interpolated_precision{
                {1,{}},
                {2,{}},
                {3,{}},
                {4,{}},
                {5,{}},
                {6,{}},
                {7,{}},
                {8,{}},
                {9,{}},
                {10,{}},
                {11,{}},
                {12,{}},
                {13,{}}
            };

        
    
        for(auto it=recAndPrec_ordered.begin();it!=recAndPrec_ordered.end();it++){
            int id=it->first;
            vector<pair<double,double>> recPrec=it->second;
            double max_prec=0;

            for(int j=0;j<recPrec.size();j++){
                max_prec=max(max_prec,recPrec[j].second);
                interpolated_precision[id].push_back(max_prec);
            }
            
        }

        double sumAP=0;
        for(auto it=interpolated_precision.begin();it!=interpolated_precision.end();it++){
            int id=it->first;
            vector<double> interPrec=it->second;
            double sum=0;

            for(int j=0;j<interPrec.size();j++){
               sum+=interPrec[j];
            }
            sumAP+=sum/interPrec.size();
            
        }

        mAP=sumAP/13;
        
    
    return mAP;
}

bool comparePairs(const std::pair<int, int>& a, const std::pair<int, int>& b) {
    // Compare the second elements of the pairs in reverse order (descending).
    return a.first > b.first;
}






  vector<map<int,Mat>> readGTboxes(){
    Mat src=imread("../GroundTruth/tray3/food_image.jpg");    //List of path
    vector<tuple<int,string>> listOfFiles{


        make_tuple(1,"../GroundTruth/tray1/bounding_boxes/food_image_bounding_box.txt"),
        make_tuple(1,"../GroundTruth/tray1/bounding_boxes/leftover1_bounding_box.txt"),
        make_tuple(1,"../GroundTruth/tray1/bounding_boxes/leftover2_bounding_box.txt"),
        
        make_tuple(2,"../GroundTruth/tray2/bounding_boxes/food_image_bounding_box.txt"),
        make_tuple(2,"../GroundTruth/tray2/bounding_boxes/leftover1_bounding_box.txt"),
        make_tuple(2,"../GroundTruth/tray2/bounding_boxes/leftover2_bounding_box.txt"),

        make_tuple(3,"../GroundTruth/tray3/bounding_boxes/food_image_bounding_box.txt"),
        make_tuple(3,"../GroundTruth/tray3/bounding_boxes/leftover1_bounding_box.txt"),
        make_tuple(3,"../GroundTruth/tray3/bounding_boxes/leftover2_bounding_box.txt"),

        make_tuple(4,"../GroundTruth/tray4/bounding_boxes/food_image_bounding_box.txt"),
        make_tuple(4,"../GroundTruth/tray4/bounding_boxes/leftover1_bounding_box.txt"),
        make_tuple(4,"../GroundTruth/tray4/bounding_boxes/leftover2_bounding_box.txt"),

        make_tuple(5,"../GroundTruth/tray5/bounding_boxes/food_image_bounding_box.txt"),
        make_tuple(5,"../GroundTruth/tray5/bounding_boxes/leftover1_bounding_box.txt"),
        make_tuple(5,"../GroundTruth/tray5/bounding_boxes/leftover2_bounding_box.txt"),

        make_tuple(6,"../GroundTruth/tray6/bounding_boxes/food_image_bounding_box.txt"),
        make_tuple(6,"../GroundTruth/tray6/bounding_boxes/leftover1_bounding_box.txt"),
        make_tuple(6,"../GroundTruth/tray6/bounding_boxes/leftover2_bounding_box.txt"),

        make_tuple(7,"../GroundTruth/tray7/bounding_boxes/food_image_bounding_box.txt"),
        make_tuple(7,"../GroundTruth/tray7/bounding_boxes/leftover1_bounding_box.txt"),
        make_tuple(7,"../GroundTruth/tray7/bounding_boxes/leftover2_bounding_box.txt"),

        make_tuple(8,"../GroundTruth/tray8/bounding_boxes/food_image_bounding_box.txt"),
        make_tuple(8,"../GroundTruth/tray8/bounding_boxes/leftover1_bounding_box.txt"),
        make_tuple(8,"../GroundTruth/tray8/bounding_boxes/leftover2_bounding_box.txt")
    };

  
    vector<map<int,Mat>>gt_boxes;
    
    for(int i=0;i<listOfFiles.size();i++){
        //cout<<" NOME FILE: "<<get<1>(listOfFiles[i])<<endl;
        ifstream fin;
        fin.open(get<1>(listOfFiles[i]));
        //fin.open("../GroundTruth/tray1/bounding_boxes/food_image_bounding_box.txt");
        if(!fin.is_open()){
            cerr<<"file does not exist"<<endl;
            exit(-1);
        }
       // cout<<"file read !"<<endl;

        
        string line;

        //read file
        vector <tuple<int,Mat>> tray;
        map<int,Mat>id_Mask;
        while(getline(fin,line)){
           vector<int> boxes_info;
           // cout<<"line: "<<line<<endl;
           
            for(int i=0;i<line.size();i++){
                char c=line[i];
                if(c!=' ' && c!='[' && c!='I' && c!='D' && c!=':'){
                    string n;
                    while(i<=line.size() && (c!=',' && c!=']' && c!='[' && c!=' ' && c!=';')){
                        n+=c;
                        i++;
                        c=line[i];
                    }
                    cout<<"num: "<<n<<endl;
                    boxes_info.push_back(stoi(n));

                }
            }
           
            Mat temp=Mat::zeros(src.size(),CV_8UC1);
            rectangle(temp,Point(boxes_info[1],boxes_info[2]),Point(boxes_info[1]+boxes_info[3],boxes_info[2]+boxes_info[4]),boxes_info[0],cv::FILLED);
            id_Mask.insert({boxes_info[0],temp});
           
        }
        gt_boxes.push_back(id_Mask);
       
    }

    //cout<<"map dimension: "<<gt_boxes.size()<<endl;
    return gt_boxes;//24 elements
}

vector<Mat> getTraysImages(){
    Mat src=imread("../GroundTruth/tray1/food_image.jpg");//only for the dimensions
    //List of path
    vector<tuple<int,string>> listOfImages{


        make_tuple(1,"../GroundTruth/tray1/food_image.jpg"),
        make_tuple(1,"../GroundTruth/tray1/leftover1.jpg"),
        make_tuple(1,"../GroundTruth/tray1/leftover2.jpg"),
        
       make_tuple(2,"../GroundTruth/tray2/food_image.jpg"),
        make_tuple(2,"../GroundTruth/tray2/leftover1.jpg"),
        make_tuple(2,"../GroundTruth/tray2/leftover2.jpg"),

        make_tuple(3,"../GroundTruth/tray3/food_image.jpg"),
        make_tuple(3,"../GroundTruth/tray3/leftover1.jpg"),
        make_tuple(3,"../GroundTruth/tray3/leftover2.jpg"),

        make_tuple(4,"../GroundTruth/tray4/food_image.jpg"),
        make_tuple(4,"../GroundTruth/tray4/leftover1.jpg"),
        make_tuple(4,"../GroundTruth/tray4/leftover2.jpg"),

        make_tuple(5,"../GroundTruth/tray5/food_image.jpg"),
        make_tuple(5,"../GroundTruth/tray5/leftover1.jpg"),
        make_tuple(5,"../GroundTruth/tray5/leftover2.jpg"),

        make_tuple(6,"../GroundTruth/tray6/food_image.jpg"),
        make_tuple(6,"../GroundTruth/tray6/leftover1.jpg"),
        make_tuple(6,"../GroundTruth/tray6/leftover2.jpg"),

        make_tuple(7,"../GroundTruth/tray7/food_image.jpg"),
        make_tuple(7,"../GroundTruth/tray7/leftover1.jpg"),
        make_tuple(7,"../GroundTruth/tray7/leftover2.jpg"),

        make_tuple(8,"../GroundTruth/tray8/food_image.jpg"),
        make_tuple(8,"../GroundTruth/tray8/leftover1.jpg"),
        make_tuple(8,"../GroundTruth/tray8/leftover2.jpg"),
    };

   
   vector<Mat> traysImages;
   
    for(int i=0;i<listOfImages.size();i++){
        Mat food_image=imread(get<1>(listOfImages[i]));
        traysImages.push_back(food_image);
    }

    return traysImages;
}





map<int,Mat> drawBoundingBoxLabel_For_metrics(vector<string> names, Mat img){
        vector<vector<Rect>> boxList=computeBoundingBox(img);
        Mat src=imread("../GroundTruth/tray1/food_image.jpg");//only for the dimensions


        std::map<std::string,int> foodClassMap{
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
   
       
        map<int,Mat> idAndBox;
    
        
    
        for( int i = 0; i < boxList.size(); i++ )
        {
            for(int j=0;j<boxList[i].size();j++){
               cout<<"Computation of boudingBoxes"<<endl;
                if(boxList[i][j].tl().x!=0 &&boxList[i][j].tl().x!=0){

                   
                    int id_img=foodClassMap[names[i]];
                    Mat temp=Mat::zeros(src.size(), CV_8U);
                    rectangle(temp, boxList[i][j].tl(), boxList[i][j].br(), id_img, cv::FILLED );
                    idAndBox.insert({id_img,temp});
                }

              
            
            }
        
            
        }
    
        return idAndBox;
  
}
