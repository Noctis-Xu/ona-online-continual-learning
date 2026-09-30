cmake_minimum_required(VERSION 3.16)
project(adaptive_sorting_bringup)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/ExecutePrimitive.srv"
  "srv/GetSortingSceneConfig.srv"
  "srv/GetSortingObservation.srv"
  "srv/PresentBinColorLabels.srv"
  "srv/PresentExperimentStatus.srv"
  "srv/PresentFeedback.srv"
  "srv/PresentObservation.srv"
)

add_executable(sorting_scene_publisher src/sorting_scene_publisher.cpp)
ament_target_dependencies(
  sorting_scene_publisher
  geometry_msgs
  moveit_msgs
  moveit_ros_planning_interface
  rclcpp
  shape_msgs
  std_msgs
  visualization_msgs
)

add_executable(sorting_primitive_executor src/sorting_primitive_executor.cpp)
ament_target_dependencies(
  sorting_primitive_executor
  geometry_msgs
  moveit_msgs
  moveit_ros_planning_interface
  rclcpp
  shape_msgs
  std_msgs
)

target_include_directories(
  sorting_scene_publisher PRIVATE include
)
target_include_directories(
  sorting_primitive_executor PRIVATE include
)

rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)
target_link_libraries(sorting_scene_publisher "${cpp_typesupport_target}")
target_link_libraries(sorting_primitive_executor "${cpp_typesupport_target}")

install(
  TARGETS sorting_scene_publisher sorting_primitive_executor
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_sorting_common test/test_sorting_common.cpp)
  target_include_directories(test_sorting_common PRIVATE include)
  ament_target_dependencies(test_sorting_common std_msgs)
endif()

ament_package()
