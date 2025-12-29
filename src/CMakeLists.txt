cmake_minimum_required(VERSION 3.0.2)
project(dogm_ros)

# Specify C++14 standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler options (e.g., optimization)
add_compile_options(-O2)

# --- Find OpenMP for parallelization ---
find_package(OpenMP REQUIRED)

# Find necessary Catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  sensor_msgs
  std_msgs
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen           
  visualization_msgs
  pcl_ros
  pcl_conversions
  message_generation
)

# Find PCL package directly
find_package(PCL 1.7 REQUIRED)

# --- Message Generation ---
add_message_files(
  FILES
  FilteredSigma.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)
# --- End Message Generation ---

# Declare Catkin package details
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp sensor_msgs std_msgs tf2_ros tf2_geometry_msgs tf2_eigen visualization_msgs pcl_ros pcl_conversions message_runtime
)

# --- Python Scripts Installation ---
catkin_install_python(PROGRAMS
  scripts/gat_transformer_node.py
  scripts/train.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY models
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
# --- End Python Installation ---

# Specify include directories
include_directories(
  include         
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# --- Define DOGM Node Executable ---
add_executable(dogm_node
  src/dogm_node.cpp
  src/dynamic_grid_map.cpp
  src/particle_filter.cpp
  src/ego_calibration.cpp
)

# Add dependencies to the custom messages
add_dependencies(dogm_node ${${PROJECT_NAME}_EXPORTED_TARGETS})

# Link DOGM node against required libraries
target_link_libraries(dogm_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  OpenMP::OpenMP_CXX
)
# --- End DOGM Node ---


# --- Define Radar Visualization Node Executable ---
add_executable(radar_velocity_viz_node
  src/radar_velocity_viz.cpp
)

# Link Radar Viz node against required libraries
target_link_libraries(radar_velocity_viz_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  OpenMP::OpenMP_CXX 
)


# --- Installation Rules ---
# Install both executables
install(TARGETS dogm_node radar_velocity_viz_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install config, launch, and rviz files
install(DIRECTORY config launch rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
# --- End Installation ---