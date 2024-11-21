/*FINAL
* @author Carlotta Schiavo,ID=2076743 and Qiu yi jian, 	ID=2085730
* 
* FoodWaste.cpp:
* Here you can Test all the function from the libraries of the project. 
* 
* HOW to TEST:
* 1. for each test you can use only a set of "before + leftover1 +leftover2 " tray. Hence if you want test
*    for example for tray 1, you have just to take away the comment from the "TRAY 1" set of images but the other set of images MUST remain commented. 
* 2. You can try the dection for a leftover at time (so take away the comments from the one that you want to try (leftover 1 or 2)).
*
* 3. BE AWARE: when you decide to test the food detection, you MUST always call firstly the food detection on the "before Tray" and than you can choose 
*    if you want to call the food Detection for leftover 1 OR leftover2. This is due to how the algorithm is thought. 
*    (Leftover food detection need the list of foods detected from the before tray).
* 4. Also the segmentation functions need to be called after the food detection hence be aware to call them after the relative food detecttion function call. 
* 
* NB: 
* 1. We have decided to add all the paths for every image in the sample in order to simplify the execution of your test
*/

#include "foodDetectionLibs.h"
#include "metricsLibs.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{

    //If you prefer to read the paths from the command line, you can take away sucj comments
    /*if (argc < 2)
    {
        cout << "Attention: Wrong number of argument" << endl;
        exit(EXIT_FAILURE);
    }*/

    // Load images
    //Mat food_image = imread(argv[1]);
    //Mat leftover1 = imread(argv[2]);
    //Mat leftover2 = imread(argv[3]);
    //Mat food_image_seg = imread(argv[1]);
    //Mat leftover_seg1 = imread(argv[2]);
    //Mat leftover_seg2 = imread(argv[3]);


    //take away the comment from the tray (JUST ONE FOR EACH EXECUTION), you want to test
    //TRAY 1
    
    Mat food_image = imread("../CV007/Food_leftover_dataset/tray1/food_image.jpg");
    Mat leftover1 = imread("../CV007/Food_leftover_dataset/tray1/leftover1.jpg");
    Mat leftover2 = imread("../CV007/Food_leftover_dataset/tray1/leftover2.jpg");
    Mat food_image_seg = imread("../CV007/Food_leftover_dataset/tray1/food_image.jpg");
    Mat leftover_seg1 = imread("../CV007/Food_leftover_dataset/tray1/leftover1.jpg");
    Mat leftover_seg2 = imread("../CV007/Food_leftover_dataset/tray1/leftover2.jpg");
    

    //TRAY 2
    /*
    Mat food_image = imread("../CV007/Food_leftover_dataset/tray2/food_image.jpg");
    Mat leftover1 = imread("../CV007/Food_leftover_dataset/tray2/leftover1.jpg");
    Mat leftover2 = imread("../CV007/Food_leftover_dataset/tray2/leftover2.jpg");
    Mat food_image_seg = imread("../CV007/Food_leftover_dataset/tray2/food_image.jpg");
    Mat leftover_seg1 = imread("../CV007/Food_leftover_dataset/tray2/leftover1.jpg");
    Mat leftover_seg2 = imread("../CV007/Food_leftover_dataset/tray2/leftover2.jpg");
    */

    //TRAY 3
    /*
    Mat food_image = imread("../CV007/Food_leftover_dataset/tray3/food_image.jpg");
    Mat leftover1 = imread("../CV007/Food_leftover_dataset/tray3/leftover1.jpg");
    Mat leftover2 = imread("../CV007/Food_leftover_dataset/tray3/leftover2.jpg");
    Mat food_image_seg = imread("../CV007/Food_leftover_dataset/tray3/food_image.jpg");
    Mat leftover_seg1 = imread("../CV007/Food_leftover_dataset/tray3/leftover1.jpg");
    Mat leftover_seg2 = imread("../CV007/Food_leftover_dataset/tray3/leftover2.jpg");
    */

    //TRAY 4
    /*
    Mat food_image = imread("../CV007/Food_leftover_dataset/tray4/food_image.jpg");
    Mat leftover1 = imread("../CV007/Food_leftover_dataset/tray4/leftover1.jpg");
    Mat leftover2 = imread("../CV007/Food_leftover_dataset/tray4/leftover2.jpg");
    Mat food_image_seg = imread("../CV007/Food_leftover_dataset/tray4/food_image.jpg");
    Mat leftover_seg1 = imread("../CV007/Food_leftover_dataset/tray4/leftover1.jpg");
    Mat leftover_seg2 = imread("../CV007/Food_leftover_dataset/tray4/leftover2.jpg");
    */

    //TRAY 5
    /*
    Mat food_image = imread("../CV007/Food_leftover_dataset/tray5/food_image.jpg");
    Mat leftover1 = imread("../CV007/Food_leftover_dataset/tray5/leftover1.jpg");
    Mat leftover2 = imread("../CV007/Food_leftover_dataset/tray5/leftover2.jpg");
    Mat food_image_seg = imread("../CV007/Food_leftover_dataset/tray5/food_image.jpg");
    Mat leftover_seg1 = imread("../CV007/Food_leftover_dataset/tray5/leftover1.jpg");
    Mat leftover_seg2 = imread("../CV007/Food_leftover_dataset/tray5/leftover2.jpg");
    */

    //TRAY 6
    /*
    Mat food_image = imread("../CV007/Food_leftover_dataset/tray6/food_image.jpg");
    Mat leftover1 = imread("../CV007/Food_leftover_dataset/tray6/leftover1.jpg");
    Mat leftover2 = imread("../CV007/Food_leftover_dataset/tray6/leftover2.jpg");
    Mat food_image_seg = imread("../CV007/Food_leftover_dataset/tray6/food_image.jpg");
    Mat leftover_seg1 = imread("../CV007/Food_leftover_dataset/tray6/leftover1.jpg");
    Mat leftover_seg2 = imread("../CV007/Food_leftover_dataset/tray6/leftover2.jpg");
    */

    //TRAY 7
    /*
    Mat food_image = imread("../CV007/Food_leftover_dataset/tray7/food_image.jpg");
    Mat leftover1 = imread("../CV007/Food_leftover_dataset/tray7/leftover1.jpg");
    Mat leftover2 = imread("../CV007/Food_leftover_dataset/tray7/leftover2.jpg");
    Mat food_image_seg = imread("../CV007/Food_leftover_dataset/tray7/food_image.jpg");
    Mat leftover_seg1 = imread("../CV007/Food_leftover_dataset/tray7/leftover1.jpg");
    Mat leftover_seg2 = imread("../CV007/Food_leftover_dataset/tray7/leftover2.jpg");
    */

    //TRAY 8
    /*
    Mat food_image = imread("../CV007/Food_leftover_dataset/tray8/food_image.jpg");
    Mat leftover1 = imread("../CV007/Food_leftover_dataset/tray8/leftover1.jpg");
    Mat leftover2 = imread("../CV007/Food_leftover_dataset/tray8/leftover2.jpg");
    Mat food_image_seg = imread("../CV007/Food_leftover_dataset/tray8/food_image.jpg");
    Mat leftover_seg1 = imread("../CV007/Food_leftover_dataset/tray8/leftover1.jpg");
    Mat leftover_seg2 = imread("../CV007/Food_leftover_dataset/tray8/leftover2.jpg");
    */




    /****************************************Food Detection**************************************************/
    /*NB:                                                                                                   */
    /*   1. The food detection on the before tray must be ALWAYS call before a leftover detection           */
    /*   2. You can call only a leftover detectio for each execution (after the call of the "before tray"   */
    /*        detection)                                                                                    */ 
    /********************************************************************************************************/

    /******Detection of the untouched tray******/
    
        vector<tuple<string, Point, double,int>> foodsOnNewTray = foodDetectionWithColor(food_image);
        vector<string> foodDetectedList;
        for (int i = 0; i < foodsOnNewTray.size(); i++)
        {
            foodDetectedList.push_back(get<0>(foodsOnNewTray[i]));
        }
    


    /*******Detection of the leftover1 tray (take awai the comment on the line above to test it)********/

    
    vector<tuple<string, Point, double, int>> leftFoods1 = leftoverDetection(leftover1, foodsOnNewTray, 1);
    


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    

    /********Detection of the leftover2 tray (take awai the comment on the line above to test it)********/

    /*
    vector<tuple<string, Point, double, int>> leftFoods2 = leftoverDetection(leftover2, foodsOnNewTray, 2);
    */

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    

   
   /****************************************Segmentation*****************************************************/
    //NB:                                                                                                   
    //   1. The food detection on the before tray must be ALWAYS call before a leftover detection           
    //   2. You can call only a leftover detectio for each execution (after the call of the "before tray"   
    //      detection)        
    //   3. After every leftover segmentation you can find also the computation of the leftover metrics, Ri.                                                                                                                                                                  */ 
    /********************************************************************************************************/

    Mat morphKernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat colorseg;

    /////////////segmentation entire tray

    
    vector<tuple<Mat, Point>> masksTray;
    masksTray = foodSegmentation(food_image_seg, foodsOnNewTray);
    Mat mascheraTray = entireMask(masksTray, food_image_seg, foodsOnNewTray);
    imshow("Before tray mask", mascheraTray);

    colorseg=foodSegmentationUtil(food_image_seg, foodsOnNewTray);
    imshow("Before tray segmentation", colorseg);
    cout<<"Please, press a keyboard key to continue. Then wait some seconds..."<<endl;
    waitKey();

   /////////////segmentation leftover 1

    
    vector<tuple<Mat, Point>> maskLeftOver1;
    maskLeftOver1 = foodSegmentation(leftover_seg1, leftFoods1);
    Mat mascheraLeftOver1 = entireMask(maskLeftOver1, leftover_seg1, leftFoods1);

    // erosion after mask
   
    morphologyEx(mascheraLeftOver1, mascheraLeftOver1, MORPH_ERODE, morphKernel, Point(-1, -1), 2);
    imshow("Leftover 1 mask", mascheraLeftOver1);

    Mat colorseg1;
    colorseg1=foodSegmentationUtil(leftover_seg1, leftFoods1);
    imshow("Leftover 1 segmentation", colorseg1);

    //////////////leftover metrics for leftover1


    double Ri = foodLeftOverMetrics(mascheraTray, mascheraLeftOver1);
    cout << "Leftover 1 metric: " << Ri << endl;

    
  
  
   /////////////segmentation left over 2

   /*
    vector<tuple<Mat, Point>> maskLeftOver2;
    maskLeftOver2 = foodSegmentation(leftover_seg2, leftFoods2);
   
    Mat mascheraLeftOver2 = entireMask(maskLeftOver2, leftover_seg2, leftFoods2);

    // erosion after mask

    morphologyEx(mascheraLeftOver2, mascheraLeftOver2, MORPH_ERODE, morphKernel, Point(-1, -1), 2);
    imshow("Leftover 2 mask", mascheraLeftOver2);


    Mat colorseg2=foodSegmentationUtil(leftover_seg2, leftFoods2);
    colorseg2=foodSegmentationUtil(leftover_seg2, leftFoods2);
    imshow("Leftover 2 segmentation", colorseg2);
    cout<<"Please, press a keyboard key to continue. Then wait some seconds..."<<endl;

    ///////////leftover metrics for leftover2

    
    double Ri = foodLeftOverMetrics(mascheraTray, mascheraLeftOver2);
    cout << "Leftover 2 metric: " << Ri << endl;
    
   
   */





    /**********************************mIoU and mAP Metrics Computation*****************************************/
    /*NB:                                                                                                      */
    /*   > GroudTruth: we use the files that you can find into the folder "GroundTruth of the project"         */
    /*   > Presictions: all the algorithms results, were saved into the folder "detected_mask" of the projects */
    /*     in order to use them for the metrics computation                                                    */
    /*   > HOW TO TEST: just take away the comments from the metrics you want to test                          */                                  
    /***********************************************************************************************************/


    /*******************mIoU************************/
    /*
    for(int i=1;i<9;i++){
        string c=to_string(i);
        Mat food_image_mask_GT = imread("../GroundTruth/tray"+c+"/masks/food_image_mask.png");
        Mat leftover1_mask_GT = imread("../GroundTruth/tray"+c+"/masks/leftover1.png");
        Mat leftover2_mask_GT = imread("../GroundTruth/tray"+c+"/masks/leftover2.png");
        Mat food_image_mask_pred = imread("../CV007/detected_masks/tray"+c+"/food_image.png");
        Mat leftover_seg1_mask_pred = imread("../CV007/detected_masks/tray"+c+"/leftover1.png");
        Mat leftover_seg2_mask_pred = imread("../CV007/detected_masks/tray"+c+"/leftover2.png");

        cout<<"TRAY "+c+":"<<endl;    
        cout<<"mIoU_before: "<<mIoU(food_image_mask_pred,food_image_mask_GT)<<endl;
        cout<<"mIoU_leftover1: "<<mIoU(food_image_mask_pred,food_image_mask_GT)<<endl;
        cout<<"mIoU_leftover2: "<<mIoU(food_image_mask_pred,food_image_mask_GT)<<endl;
        
    }
    */




    /*********mAP*********************************/

     
  
    //cout<<"mAP: "<<mAP();

    waitKey(0);
    return 0;
}