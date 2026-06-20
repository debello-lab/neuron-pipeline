from mcf2swc import MeshManager, SkeletonGraph, SkeletonOptimizer, SkeletonOptimizerOptions

# Load mesh and skeleton
mesh_mgr = MeshManager(mesh_path="mesh.obj")
skeleton = SkeletonGraph.from_txt("skeleton.polylines.txt")

# Configure optimization
opts = SkeletonOptimizerOptions(
    max_iterations=100,
    step_size=0.1,
    convergence_threshold=1e-4,
    preserve_terminal_nodes=True,  # Preserve endpoints
    preserve_branch_nodes=False,   # Allow branch nodes to move
    smoothing_weight=0.5,
    n_rays=6,  # Number of rays for medial axis estimation
    verbose=True
)

# Optimize skeleton
optimizer = SkeletonOptimizer(skeleton, mesh_mgr.mesh, opts)
optimized_skeleton = optimizer.optimize()

# Save optimized skeleton (GraphML format)
optimized_skeleton.to_txt("skeleton_optimized.graphml")
