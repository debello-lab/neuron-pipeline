#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_count_stop_predicate.h>
#include <CGAL/IO/OBJ.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <iostream>
#include <string>
#include <vector>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Surface_mesh;

namespace SMS = CGAL::Surface_mesh_simplification;
namespace PMP = CGAL::Polygon_mesh_processing;

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <input.obj> <output.obj> <target>" << std::endl;
        std::cerr << "  <target> can be:" << std::endl;
        std::cerr << "    - Integer: target edge count (e.g., 1000)" << std::endl;
        std::cerr << "    - Float 0-1: fraction of original edges (e.g., 0.5)" << std::endl;
        std::cerr << "Examples:" << std::endl;
        std::cerr << "  " << argv[0] << " data/input.obj data/output.obj 1000" << std::endl;
        std::cerr << "  " << argv[0] << " data/input.obj data/output.obj 0.5" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    std::string target_arg = argv[3];

    Surface_mesh mesh;

    // Read OBJ file as polygon soup for better compatibility
    std::vector<Point_3> points;
    std::vector<std::vector<std::size_t>> polygons;

    if (!CGAL::IO::read_OBJ(input_file, points, polygons))
    {
        std::cerr << "Error: Cannot read input file: " << input_file << std::endl;
        return 1;
    }

    // Repair and convert to mesh
    PMP::repair_polygon_soup(points, polygons);
    PMP::orient_polygon_soup(points, polygons);
    PMP::polygon_soup_to_polygon_mesh(points, polygons, mesh);

    std::cout << "Input mesh: " << mesh.number_of_vertices() << " vertices, "
              << mesh.number_of_edges() << " edges, "
              << mesh.number_of_faces() << " faces" << std::endl;

    // Determine if target is edge count or fraction
    int target_edge_count;
    double target_value = std::stod(target_arg);

    if (target_value > 0 && target_value < 1)
    {
        // Fraction mode
        target_edge_count = static_cast<int>(mesh.number_of_edges() * target_value);
        std::cout << "Target fraction: " << target_value << " (" << target_edge_count << " edges)" << std::endl;
    }
    else if (target_value >= 1)
    {
        // Edge count mode
        target_edge_count = static_cast<int>(target_value);
        std::cout << "Target edge count: " << target_edge_count << std::endl;
    }
    else
    {
        std::cerr << "Error: Target must be either >= 1 (edge count) or between 0 and 1 (fraction)" << std::endl;
        return 1;
    }

    SMS::Edge_count_stop_predicate<Surface_mesh> stop(target_edge_count);

    std::cout << "Simplifying mesh to " << target_edge_count << " edges..." << std::endl;
    int removed_edges = SMS::edge_collapse(mesh, stop);

    std::cout << "Removed " << removed_edges << " edges" << std::endl;
    std::cout << "Output mesh: " << mesh.number_of_vertices() << " vertices, "
              << mesh.number_of_edges() << " edges, "
              << mesh.number_of_faces() << " faces" << std::endl;

    if (!CGAL::IO::write_polygon_mesh(output_file, mesh))
    {
        std::cerr << "Error: Cannot write output file: " << output_file << std::endl;
        return 1;
    }

    std::cout << "Mesh saved to: " << output_file << std::endl;
    return 0;
}