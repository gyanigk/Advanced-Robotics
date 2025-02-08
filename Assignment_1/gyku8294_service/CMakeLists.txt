cmake_minimum_required(VERSION 3.8)
project(gyku8294_service)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)  
find_package(rosidl_default_generators REQUIRED)

# Generate service files
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/Gyku8294Service.srv"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)  # Remove this when adding copyright
  set(ament_cmake_cpplint_FOUND TRUE)    # Remove this when pushing to Git
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
