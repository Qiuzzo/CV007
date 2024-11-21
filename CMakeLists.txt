cmake_minimum_required(VERSION 2.7)

project(foodWaste)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(foodWasteLibs 
                CV007/foodSegBoxLibs.h 
                CV007/foodSegBoxLibs.cpp 
                CV007/foodDetectionLibs.h 
                CV007/foodDetectionLibs.cpp
                CV007/metricsLibs.h 
                CV007/metricsLibs.cpp)
                
add_executable(foodWaste CV007/foodWaste.cpp)

target_link_libraries(foodWaste ${OpenCV_LIBS} foodWasteLibs)

