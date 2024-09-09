#--------------------------------------------------------------------------------------------------------------------------#
#Project configuration
#--------------------------------------------------------------------------------------------------------------------------#
cmake_minimum_required(VERSION 3.16)
project(pi_camera_stabilization DESCRIPTION "pi_camera_stabilization")


find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
#find_package(Boost REQUIRED COMPONENTS system program_options CONFIG) 

#--------------------------------------------------------------------------------------------------------------------------#
#Executable configuration
#--------------------------------------------------------------------------------------------------------------------------#
add_executable(${PROJECT_NAME}
    src/pi_camera_stabilization.cpp
)

target_compile_features(${PROJECT_NAME} 
	PRIVATE cxx_std_17
)

target_link_libraries(${PROJECT_NAME}
 ${OpenCV_LIBS} 
)

#include(CMakePrintHelpers)
#cmake_print_variables(CMAKE_SYSTEM_PROCESSOR)
#target_link_directories(${PROJECT_NAME} 
#	PRIVATE lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu
#)

#target_include_directories(${PROJECT_NAME} 
#	PRIVATE include
#)


#--------------------------------------------------------------------------------------------------------------------------#
#Install
#--------------------------------------------------------------------------------------------------------------------------#
install(TARGETS ${PROJECT_NAME} DESTINATION bin)