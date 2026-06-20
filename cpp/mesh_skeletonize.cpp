#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/OBJ.h>
#include <CGAL/Mean_curvature_flow_skeletonization.h>
#include <CGAL/boost/graph/split_graph_into_polylines.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/boost/graph/helpers.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using Surface_mesh = CGAL::Surface_mesh<Point_3>;
using Skeletonization = CGAL::Mean_curvature_flow_skeletonization<Surface_mesh>;
using Skeleton = Skeletonization::Skeleton;
using Skeleton_vertex = Skeleton::vertex_descriptor;

namespace PMP = CGAL::Polygon_mesh_processing;

struct Polyline_writer {
    const Skeleton& skeleton;
    std::ofstream& out;
    int polyline_size;
    std::stringstream sstr;

    Polyline_writer(const Skeleton& skeleton_ref, std::ofstream& out_ref)
        : skeleton(skeleton_ref), out(out_ref), polyline_size(0) {}

    void start_new_polyline() {
        polyline_size = 0;
        sstr.str("");
        sstr.clear();
    }

    void add_node(Skeleton_vertex v) {
        const Point_3& point = skeleton[v].point;
        ++polyline_size;
        sstr << ' ' << point.x() << ' ' << point.y() << ' ' << point.z();
    }

    void end_polyline() {
        out << polyline_size << sstr.str() << '\n';
    }
};

int main(int argc, char** argv) {
    if (argc != 3 && argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.obj> <output.polylines.txt> [w_H] [w_M]"
                  << std::endl;
        return 1;
    }

    const std::string input_file = argv[1];
    const std::string output_file = argv[2];
    const double quality_speed_tradeoff =
        argc >= 4 ? std::stod(argv[3]) : 0.5;
    const double medially_centered_speed_tradeoff =
        argc >= 5 ? std::stod(argv[4]) : 5.0;

    std::vector<Point_3> points;
    std::vector<std::vector<std::size_t>> polygons;
    if (!CGAL::IO::read_OBJ(input_file, points, polygons)) {
        std::cerr << "Error: Cannot read OBJ file: " << input_file << std::endl;
        return 1;
    }

    PMP::repair_polygon_soup(points, polygons);
    PMP::orient_polygon_soup(points, polygons);

    Surface_mesh mesh;
    PMP::polygon_soup_to_polygon_mesh(points, polygons, mesh);

    if (!CGAL::is_triangle_mesh(mesh)) {
        std::cerr << "Error: Input mesh is not a triangle mesh" << std::endl;
        return 1;
    }

    if (mesh.number_of_vertices() == 0 || mesh.number_of_faces() == 0) {
        std::cerr << "Error: Input mesh is empty" << std::endl;
        return 1;
    }

    if (!CGAL::is_closed(mesh)) {
        std::cerr << "Error: Skeletonization requires a closed mesh" << std::endl;
        return 1;
    }

    Skeleton skeleton;
    Skeletonization skeletonization(mesh);
    skeletonization.set_quality_speed_tradeoff(quality_speed_tradeoff);
    skeletonization.set_is_medially_centered(true);
    skeletonization.set_medially_centered_speed_tradeoff(
        medially_centered_speed_tradeoff
    );
    skeletonization(skeleton);

    std::ofstream output(output_file);
    if (!output) {
        std::cerr << "Error: Cannot open output file: " << output_file
                  << std::endl;
        return 1;
    }

    Polyline_writer writer(skeleton, output);
    CGAL::split_graph_into_polylines(skeleton, writer);
    output.close();

    std::cout << "Input mesh: " << mesh.number_of_vertices() << " vertices, "
              << mesh.number_of_faces() << " faces" << std::endl;
    std::cout << "Skeleton vertices: " << boost::num_vertices(skeleton)
              << std::endl;
    std::cout << "Skeleton edges: " << boost::num_edges(skeleton)
              << std::endl;
    std::cout << "quality_speed_tradeoff=" << quality_speed_tradeoff
              << std::endl;
    std::cout << "medially_centered_speed_tradeoff="
              << medially_centered_speed_tradeoff << std::endl;
    std::cout << "Polylines written to: " << output_file << std::endl;
    return 0;
}