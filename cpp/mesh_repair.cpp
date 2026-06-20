#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/IO/OBJ.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/shape_predicates.h>
#include <iostream>
#include <string>
#include <vector>
#include <set>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Surface_mesh;
typedef Surface_mesh::Vertex_index Vertex_index;
typedef Surface_mesh::Face_index Face_index;
typedef Surface_mesh::Halfedge_index Halfedge_index;

namespace PMP = CGAL::Polygon_mesh_processing;

struct RepairStats
{
    int initial_vertices = 0;
    int initial_faces = 0;
    int initial_components = 0;
    int components_removed = 0;
    bool had_self_intersections = false;
    int holes_filled = 0;
    int degenerate_faces_removed = 0;
    int final_vertices = 0;
    int final_faces = 0;
};

void print_stats(const RepairStats &stats)
{
    std::cout << "\n=== Repair Summary ===" << std::endl;
    std::cout << "Initial: " << stats.initial_vertices << " vertices, "
              << stats.initial_faces << " faces, "
              << stats.initial_components << " components" << std::endl;

    if (stats.components_removed > 0)
    {
        std::cout << "Removed " << stats.components_removed << " small components" << std::endl;
    }

    if (stats.had_self_intersections)
    {
        std::cout << "Detected and reported self-intersections" << std::endl;
    }

    if (stats.holes_filled > 0)
    {
        std::cout << "Filled " << stats.holes_filled << " boundary holes" << std::endl;
    }

    if (stats.degenerate_faces_removed > 0)
    {
        std::cout << "Removed " << stats.degenerate_faces_removed << " degenerate faces" << std::endl;
    }

    std::cout << "Final: " << stats.final_vertices << " vertices, "
              << stats.final_faces << " faces" << std::endl;
    std::cout << "======================" << std::endl;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input.obj> <output.obj>" << std::endl;
        std::cerr << "Repairs mesh to ensure:" << std::endl;
        std::cerr << "  1. Single connected component (keeps largest)" << std::endl;
        std::cerr << "  2. No self-intersections (detected and reported)" << std::endl;
        std::cerr << "  3. No boundary holes (all filled)" << std::endl;
        std::cerr << "Example: " << argv[0] << " data/input.obj data/output_repaired.obj" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    Surface_mesh mesh;
    RepairStats stats;

    // Read input mesh using OBJ-specific function
    std::cout << "Reading mesh from: " << input_file << std::endl;

    // Read as polygon soup first
    std::vector<Point_3> points;
    std::vector<std::vector<std::size_t>> polygons;

    if (!CGAL::IO::read_OBJ(input_file, points, polygons))
    {
        std::cerr << "Error: Cannot read OBJ file: " << input_file << std::endl;
        return 1;
    }

    std::cout << "Read polygon soup: " << points.size() << " points, "
              << polygons.size() << " polygons" << std::endl;

    // Repair and orient the polygon soup
    PMP::repair_polygon_soup(points, polygons);
    PMP::orient_polygon_soup(points, polygons);

    // Convert to polygon mesh
    PMP::polygon_soup_to_polygon_mesh(points, polygons, mesh);

    stats.initial_vertices = mesh.number_of_vertices();
    stats.initial_faces = mesh.number_of_faces();

    std::cout << "Input mesh: " << stats.initial_vertices << " vertices, "
              << stats.initial_faces << " faces" << std::endl;

    // ========================================
    // STEP 1: PRE-PROCESSING
    // ========================================
    std::cout << "\n[1/5] Pre-processing..." << std::endl;

    // Stitch borders to merge duplicate boundaries
    std::cout << "  - Stitching borders..." << std::endl;
    PMP::stitch_borders(mesh);

    // Duplicate non-manifold vertices
    std::cout << "  - Checking for non-manifold vertices..." << std::endl;
    std::size_t duplicated = PMP::duplicate_non_manifold_vertices(mesh);
    if (duplicated > 0)
    {
        std::cout << "    Fixed " << duplicated << " non-manifold vertices" << std::endl;
    }

    // ========================================
    // STEP 2: COMPONENT SELECTION
    // ========================================
    std::cout << "\n[2/5] Selecting largest connected component..." << std::endl;

    // Count connected components
    Surface_mesh::Property_map<Face_index, std::size_t> fccmap =
        mesh.add_property_map<Face_index, std::size_t>("f:CC").first;
    stats.initial_components = PMP::connected_components(mesh, fccmap);

    std::cout << "  Found " << stats.initial_components << " connected component(s)" << std::endl;

    if (stats.initial_components > 1)
    {
        std::cout << "  Keeping only the largest component..." << std::endl;
        PMP::keep_largest_connected_components(mesh, 1);
        stats.components_removed = stats.initial_components - 1;
        std::cout << "  Removed " << stats.components_removed << " component(s)" << std::endl;
    }

    // ========================================
    // STEP 3: SELF-INTERSECTION DETECTION
    // ========================================
    std::cout << "\n[3/5] Checking for self-intersections..." << std::endl;

    bool has_self_intersections = PMP::does_self_intersect(mesh);

    if (has_self_intersections)
    {
        std::cout << "  WARNING: Mesh has self-intersections!" << std::endl;

        // Report intersecting face pairs
        std::vector<std::pair<Face_index, Face_index>> intersected_faces;
        PMP::self_intersections(mesh, std::back_inserter(intersected_faces));
        std::cout << "  Found " << intersected_faces.size() << " pairs of intersecting faces" << std::endl;

        stats.had_self_intersections = true;

        std::cout << "  NOTE: Self-intersection resolution requires manual intervention or" << std::endl;
        std::cout << "        specialized tools. Continuing with other repairs..." << std::endl;
    }
    else
    {
        std::cout << "  No self-intersections detected" << std::endl;
    }

    // ========================================
    // STEP 4: BOUNDARY FILLING
    // ========================================
    std::cout << "\n[4/5] Filling boundary holes..." << std::endl;

    // Collect all border halfedges
    std::vector<Halfedge_index> border_cycles;
    for (Halfedge_index h : mesh.halfedges())
    {
        if (CGAL::is_border(h, mesh))
        {
            border_cycles.push_back(h);
        }
    }

    if (border_cycles.empty())
    {
        std::cout << "  No boundary holes found - mesh is already closed" << std::endl;
    }
    else
    {
        std::cout << "  Found boundary halfedges, identifying holes..." << std::endl;

        // Track which halfedges we've already processed
        std::set<Halfedge_index> processed;

        for (Halfedge_index h : border_cycles)
        {
            if (processed.find(h) != processed.end())
            {
                continue; // Already filled this cycle
            }

            // Mark all halfedges in this cycle as processed
            Halfedge_index current = h;
            do
            {
                processed.insert(current);
                current = next(current, mesh);
            } while (current != h);

            // Fill the hole
            std::cout << "  Filling hole..." << std::endl;
            std::vector<Face_index> patch_faces;
            PMP::triangulate_hole(mesh, h, std::back_inserter(patch_faces));

            stats.holes_filled++;
            std::cout << "    Added " << patch_faces.size() << " faces" << std::endl;
        }

        std::cout << "  Filled " << stats.holes_filled << " hole(s)" << std::endl;
    }

    // Verify mesh is now closed
    bool is_closed = true;
    for (Halfedge_index h : mesh.halfedges())
    {
        if (CGAL::is_border(h, mesh))
        {
            is_closed = false;
            break;
        }
    }

    if (!is_closed)
    {
        std::cout << "  WARNING: Mesh still has boundaries after hole filling!" << std::endl;
    }
    else
    {
        std::cout << "  Mesh is now closed (no boundaries)" << std::endl;
    }

    // ========================================
    // STEP 5: POST-PROCESSING
    // ========================================
    std::cout << "\n[5/5] Post-processing..." << std::endl;

    // Remove degenerate faces
    std::cout << "  - Removing degenerate faces..." << std::endl;
    stats.degenerate_faces_removed = PMP::remove_degenerate_faces(mesh);
    if (stats.degenerate_faces_removed > 0)
    {
        std::cout << "    Removed " << stats.degenerate_faces_removed << " degenerate face(s)" << std::endl;
    }

    // Collect garbage
    mesh.collect_garbage();

    stats.final_vertices = mesh.number_of_vertices();
    stats.final_faces = mesh.number_of_faces();

    // ========================================
    // VALIDATION
    // ========================================
    std::cout << "\n[Validation]" << std::endl;

    bool is_valid = CGAL::is_valid_polygon_mesh(mesh);
    std::cout << "  Valid polygon mesh: " << (is_valid ? "YES" : "NO") << std::endl;

    bool is_triangle = CGAL::is_triangle_mesh(mesh);
    std::cout << "  Triangle mesh: " << (is_triangle ? "YES" : "NO") << std::endl;

    std::cout << "  Closed (no boundaries): " << (is_closed ? "YES" : "NO") << std::endl;

    std::cout << "  Single component: YES (enforced)" << std::endl;

    std::cout << "  Self-intersections: " << (has_self_intersections ? "PRESENT (see warnings)" : "NONE") << std::endl;

    // Write output
    std::cout << "\nWriting repaired mesh to: " << output_file << std::endl;
    if (!CGAL::IO::write_polygon_mesh(output_file, mesh))
    {
        std::cerr << "Error: Cannot write output file: " << output_file << std::endl;
        return 1;
    }

    print_stats(stats);

    if (!is_valid || !is_triangle || !is_closed || has_self_intersections)
    {
        std::cout << "\nWARNING: Repaired mesh may still have issues. Review validation results above." << std::endl;
        return 1;
    }

    std::cout << "\nMesh repair completed successfully!" << std::endl;
    return 0;
}