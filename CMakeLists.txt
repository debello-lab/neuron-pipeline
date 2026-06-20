cmake_minimum_required(VERSION 3.20)
project(neuromesh-cgal)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CGAL REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(mesh_skeletonize cpp/mesh_skeletonize.cpp)
target_link_libraries(mesh_skeletonize PRIVATE CGAL::CGAL Eigen3::Eigen)
target_compile_definitions(mesh_skeletonize PRIVATE CGAL_EIGEN3_ENABLED)

add_executable(mesh_repair cpp/mesh_repair.cpp)
target_link_libraries(mesh_repair PRIVATE CGAL::CGAL Eigen3::Eigen)
target_compile_definitions(mesh_repair PRIVATE CGAL_EIGEN3_ENABLED)

add_executable(mesh_simplify cpp/mesh_simplify.cpp)
target_link_libraries(mesh_simplify PRIVATE CGAL::CGAL Eigen3::Eigen)
target_compile_definitions(mesh_simplify PRIVATE CGAL_EIGEN3_ENABLED)
