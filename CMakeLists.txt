cmake_minimum_required(VERSION 2.6)
project(homeworks8_4_2)
 
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
 
#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
FIND_PACKAGE(RTABMap REQUIRED)
find_package(OpenCV REQUIRED)
FIND_PACKAGE(G2O REQUIRED)
include_directories(${RTABMap_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${G2O_INCLUDE_DIR})
 
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
 
 
add_executable(homeworks8_4_2 man.cpp)
 
target_link_libraries(homeworks8_4_2 ${OpenCV_LIBS} ${CERES_LIBRARIES} ${RTABMap_LIBRARIES} ${G2O_LIBS})
 
