/***********PROJECT STRUCTURE DESCRIPTION****************/

Inside you can find the following folders structure:


- "CV007":
	FOLDERS:
	-"segmented_sample": contains all the sample of the food used for the food detection;
	-"Food_leftover_dataset":the dataset give to us for the exam;
	-"detected_mask": it contains the segmentation for each tray (food_image, leftover1, leftover2). These
			   masks are mainly used for the measurements.
	FILES:
	- foodWaste.cpp -> the executable file.
	- foodDetectionLibs.h -> libraries for food Detection
	- foodDetectionLibs.cpp -> implemenetation of the libraries for food detection
	- foodSegBoxLibs.h -> libraries for food Segmentation
	- foodSegBoxLibs.cpp -> implemenetation of the libraries for food Segmentation
	- MetricsLibs.h -> libraries for the metrics
	- MetricsLibs.cpp -> implemenetation of the libraries for the metrics
	
 - "GroundTruth":
	It contains all the samples use in order to do the tests. 
 
 - "CMakeLists.txt"
 - "build" folder for cmake
	

	

	
	

