"""
Voxel and Mesh Cleaning Operations
Neural Reconstruction Pipeline

This module provides cleaning operations for both voxel masks and surface meshes:
- Remove noise components (specks from accidental clicks)
- Fill internal holes
- Smooth surfaces
- Remove spurious branches
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Literal
from skimage.measure import label
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation, binary_closing
import logging
from dataclasses import dataclass


def _fill_holes_multiaxis(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes by applying binary_fill_holes along each of the three axes
    independently and taking the union of all three results with the original.

    Rationale: scipy's binary_fill_holes requires a void to be fully enclosed
    in 3D.  Elongated internal voids (e.g. a gap running along Z) may be open
    at their ends in 3D -- not enclosed -- but appear as closed holes in every
    XY cross-section.  Filling per-slice catches these cases.

    This is intentionally more aggressive than a 3D-only fill.  Use the
    `fill_holes_per_axis` flag in clean_mask to opt in.

    Args:
        mask: Boolean 3D array (Z, Y, X).

    Returns:
        Boolean 3D array with axis-wise holes filled (superset of input).
    """
    result = mask.copy()
    # Fill along Z axis (each XY slice independently)
    for z in range(mask.shape[0]):
        result[z] |= binary_fill_holes(mask[z])
    # Fill along Y axis (each XZ slice independently)
    for y in range(mask.shape[1]):
        result[:, y, :] |= binary_fill_holes(mask[:, y, :])
    # Fill along X axis (each YZ slice independently)
    for x in range(mask.shape[2]):
        result[:, :, x] |= binary_fill_holes(mask[:, :, x])
    return result


def _make_anisotropic_structuring_element(
    radius_xy: float,
    voxel_size_um: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    """
    Build a 3D ellipsoidal structuring element scaled to physical voxel dimensions.

    When voxels are anisotropic (e.g. 5 nm XY vs 50 nm Z), a ball of radius N
    voxels is physically much larger in Z than in XY.  This element has the same
    physical radius in all directions, so closing bridges gaps of equal physical
    size regardless of axis.

    Args:
        radius_xy  : Closing radius in XY voxels.
        voxel_size_um : (sx, sy, sz) voxel dimensions in microns.
                        If None an isotropic ball is returned.

    Returns:
        Boolean 3D structuring element array (Z, Y, X).
    """
    if voxel_size_um is None:
        from skimage.morphology import ball
        return ball(radius_xy).astype(bool)

    sx, sy, sz = voxel_size_um
    # Physical radius = radius_xy * sx  (assumes sx == sy)
    phys_radius = radius_xy * sx

    # How many voxels does that radius correspond to in each axis?
    rx = int(np.ceil(phys_radius / sx))
    ry = int(np.ceil(phys_radius / sy))
    rz = max(1, int(np.ceil(phys_radius / sz)))

    # Ellipsoid: (x/rx)^2 + (y/ry)^2 + (z/rz)^2 <= 1
    zz, yy, xx = np.mgrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
    element = ((xx / rx) ** 2 + (yy / ry) ** 2 + (zz / rz) ** 2) <= 1.0
    return element.astype(bool)



# Explicit coordinate frame types
CoordFrame = Literal[
    'vast_global_voxel_xyz',      # VAST dataset space, corner-of-voxel, (x,y,z) ordering
    'local_bbox_voxel_xyz',       # Bounding-box-relative, corner-of-voxel, (x,y,z)
    'numpy_array_index_zyx',      # NumPy array indices, (z,y,x) ordering
    'physical_um_xyz_center',     # Physical micrometers, center-of-voxel, (x,y,z)
]

@dataclass
class VoxelData:
    """Voxel mask with validated spatial metadata."""
    
    # The mask itself (NumPy array index space)
    mask: np.ndarray  # shape (Z, Y, X), dtype bool
    
    # Spatial reference (how to convert mask indices to physical space)
    bbox_min_vox: Tuple[int, int, int]        # (minx, miny, minz) in VAST global voxel space
    voxel_size_um: Tuple[float, float, float] # (sx, sy, sz) -- voxel dimensions in µm
    
    # Explicit coordinate frame labels
    mask_index_frame: CoordFrame = 'numpy_array_index_zyx'
    bbox_frame: CoordFrame = 'vast_global_voxel_xyz'
    
    # Quality metrics
    total_voxels: int = 0
    bbox_volume_voxels: int = 0
    fill_fraction: float = 0.0  # total_voxels / bbox_volume
    
    def __post_init__(self):
        """Validate internal consistency."""
        # Shape checks
        assert self.mask.ndim == 3, f"Mask must be 3D, got shape {self.mask.shape}"
        assert self.mask.dtype == bool, f"Mask must be boolean, got {self.mask.dtype}"
        
        # Coordinate tuple lengths
        assert len(self.bbox_min_vox) == 3, "bbox_min_vox must be (x, y, z)"
        assert len(self.voxel_size_um) == 3, "voxel_size_um must be (sx, sy, sz)"
        
        # Positivity checks
        assert all(s > 0 for s in self.voxel_size_um), \
            f"Voxel sizes must be positive, got {self.voxel_size_um}"
        
        # Compute quality metrics
        self.total_voxels = int(np.sum(self.mask))
        Z, Y, X = self.mask.shape
        self.bbox_volume_voxels = X * Y * Z
        self.fill_fraction = self.total_voxels / self.bbox_volume_voxels if self.bbox_volume_voxels > 0 else 0.0
        
        # Sanity checks for degenerate cases
        if self.total_voxels == 0:
            raise ValueError("Extracted mask is completely empty")
        
        # Fill-fraction is not a reliable check for elongated morphologies
        # (a thin axon in a large bounding box can be << 0.1% filled and still
        # be a valid extraction).  Use an absolute voxel floor instead.
        if self.total_voxels < 1000:
            raise ValueError(
                f"Mask is suspiciously sparse: only {self.total_voxels} voxels filled "
                f"({self.fill_fraction*100:.4f}% of bbox). This likely indicates an extraction error."
            )
    
    def to_physical_um(self, z_idx: int, y_idx: int, x_idx: int, 
                      use_voxel_centers: bool = True) -> Tuple[float, float, float]:
        """
        Convert NumPy array indices to physical micrometers.
        
        Args:
            z_idx, y_idx, x_idx: Indices in the mask array (ZYX ordering)
            use_voxel_centers: If True, add 0.5 to place point at voxel center
                              If False, use corner-of-voxel convention
        
        Returns:
            (x_um, y_um, z_um) in physical space
        """
        # Step 1: Array index -> local voxel coordinate
        # (add 0.5 for center-of-voxel convention)
        offset = 0.5 if use_voxel_centers else 0.0
        local_x = x_idx + offset
        local_y = y_idx + offset
        local_z = z_idx + offset
        
        # Step 2: Local voxel -> global voxel (add bbox origin)
        global_x_vox = local_x + self.bbox_min_vox[0]
        global_y_vox = local_y + self.bbox_min_vox[1]
        global_z_vox = local_z + self.bbox_min_vox[2]
        
        # Step 3: Global voxel -> physical µm (scale by voxel size)
        x_um = global_x_vox * self.voxel_size_um[0]
        y_um = global_y_vox * self.voxel_size_um[1]
        z_um = global_z_vox * self.voxel_size_um[2]
        
        return (x_um, y_um, z_um)

class VoxelCleaner:
    """
    Clean binary voxel masks for skeletonization.

    Removes artifacts that would corrupt skeleton quality:
    - Small disconnected components (specks from annotation errors)
    - Internal holes and voids caused by incomplete segmentation fill
    - Surface roughness (optional smoothing)

    Pipeline order
    --------------
    1. Morphological closing  -- bridges annotation gaps so that interior
       voids become truly enclosed (not connected to the exterior).
    2. Component filtering    -- discard small specks / keep largest.
    3. Hole filling           -- seal now-enclosed interior voids.
    4. Smoothing              -- optional surface regularisation.

    Recommended parameters for barn owl auditory cortex data
    ---------------------------------------------------------
    Gap source: incomplete boundary mask + fill workflow produces surface
    discontinuities of 0.6-1.1 µm (occasionally up to ~2 µm).
    Internal dark structures (mitochondria, vesicles) are not biology to
    preserve -- the entire neuron interior should be solid.

    Recommended settings:
      closing_radius_um = 0.05   # bridges typical 0.6-1.1 µm gaps
      keep_largest_only = True
      fill_holes        = True

    Run diag_cleaning_param_sweep.py to validate the chosen radius against
    your specific segment types before committing to a value.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def clean_mask(
        self,
        mask: np.ndarray,
        voxel_size_um: Tuple[float, float, float], # For biological reasoning, different mips different voxel sizes, so this is required to choose an appropriate closing structuring element size
        bbox_min_vox: Tuple[int, int, int] = (0, 0, 0),
        min_component_voxels: int = 5000,
        keep_largest_only: bool = True,
        fill_holes: bool = True,
        fill_holes_per_axis: bool = False,
        smooth_iterations: int = 0,
        closing_radius_um: float = 0.05, # Micrometers
        closing_iterations: int = 1,
    ) -> Tuple[VoxelData, Dict[str, Any]]:
        """
        Clean a binary voxel mask.

        Args:
            mask                 : Boolean 3D array (Z, Y, X).
            min_component_voxels : Minimum component size to keep when
                                   keep_largest_only=False.
            keep_largest_only    : Keep only the largest connected component.
            fill_holes           : Fill internal holes / voids after closing
                                   using a 3D flood-fill from the exterior.
                                   Only seals voids that are fully enclosed in
                                   3D after the closing step.
            fill_holes_per_axis  : Additionally fill holes in each 2D slice
                                   along Z, Y, and X and union the results.
                                   More aggressive than the 3D fill alone --
                                   catches elongated voids that are open-ended
                                   in 3D (e.g. a gap running the length of a
                                   dendrite) but appear closed in cross-section.
                                   Requires fill_holes=True.  Safe to enable for
                                   messy segmentations; may over-fill on very
                                   sparse or C-shaped structures.
            smooth_iterations    : Morphological open/close smoothing passes
                                   (0 = disabled).
            closing_radius_um    : Physical radius for the morphological closing
                                   step in micrometers (0 = skip closing).
                                   Bridges surface discontinuities smaller than
                                   this radius.  Recommended starting point for
                                   barn owl auditory cortex data: 0.05 µm, which
                                   covers the typical gap range of 0.6-1.1 µm
                                   from incomplete boundary mask fills.  Increase
                                   to ~1.5 µm for the occasional ~2 µm gap.
                                   Use diag_cleaning_param_sweep.py to tune.
            closing_iterations   : Number of closing passes to apply.  A single
                                   pass bridges gaps up to ~radius_um.  Multiple
                                   passes with the same element progressively
                                   seal more complex gap patterns and are cheaper
                                   than one very large element.  Typical range:
                                   1 (default) to 3 for messy inputs.
            voxel_size_um        : (sx, sy, sz) voxel dimensions in microns.
                                   When supplied, the closing structuring element
                                   is scaled to be isotropic in physical space,
                                   which is important for anisotropic datasets
                                   (e.g. 5 nm XY vs 50 nm Z).

        Returns:
            Tuple of (cleaned_mask, stats)

        Tuning guide for messy segmentations
        -------------------------------------
        Internal gaps (voids inside the neuron body):
          • Increase closing_radius_um so the shell gaps get bridged before
            hole filling (start at 2-4 XY voxels).
          • Enable fill_holes_per_axis=True to catch elongated open voids that
            the 3D filler misses.
          • Use closing_iterations=2 or 3 if a single pass is insufficient.

        Borders that do not touch (surface discontinuities):
          • Increase closing_radius_um -- the closing diameter must exceed the
            physical gap width.
          • Increase closing_iterations to reach gaps through narrow passages.
          • After cleaning, inspect stats['closing_voxels_added'] per iteration
            to judge whether additional passes are still contributing.
        """
        self.logger.info("[clean_masks] Starting mask cleaning...")

        original_voxel_count = int(np.sum(mask))

        stats = {
            'original_voxel_count': original_voxel_count,
            'original_components': 0,
            'kept_components': 0,
            'removed_components': 0,
            'closing_voxels_added': 0,
            'closing_iterations_run': 0,
            'holes_filled': False,
            'hole_fill_voxels_added': 0,
            'per_axis_fill_voxels_added': 0,
            'smoothing_iterations': smooth_iterations,
            'final_voxel_count': 0,
            'voxels_added': 0,
            'voxels_removed': 0,
        }

        # ------------------------------------------------------------------
        # Step 1: Morphological closing (iterated)
        #
        # Bridges gaps left by incomplete annotation fill so that interior
        # voids are enclosed (not connected to the exterior) before we call
        # binary_fill_holes.  Without this step, gaps in the shell connect
        # interior voids to the background and fill_holes does nothing.
        #
        # A physically-isotropic ellipsoidal structuring element is used so
        # that the closing distance is the same in XY and Z regardless of
        # voxel anisotropy.
        #
        # Multiple iterations bridge progressively more complex gap patterns;
        # each pass operates on the output of the previous one, so iteration N
        # can reach gaps that were blocked by unfilled voids in iteration N-1.
        # The original voxels are OR-ed back in after every pass to prevent the
        # dilation half of closing from erasing true signal at the boundary.
        # ------------------------------------------------------------------
        cleaned_mask = mask.astype(bool)

        # Convert um to voxels
        sx, sy, sz = voxel_size_um
        avg_voxel_um = (sx + sy) / 2.0
        closing_radius_vox = max(1, int(np.round(closing_radius_um / avg_voxel_um)))

        self.logger.info(
            f"[clean_mask] Closing radius: {closing_radius_um:.2f} µm "
            f"(~{closing_radius_vox} voxels at {avg_voxel_um*1000:.1f} nm/voxel)"
        )

        if closing_radius_um > 0 and closing_iterations > 0:
            struct = _make_anisotropic_structuring_element(
                closing_radius_vox, voxel_size_um
            )
            self.logger.info(
                f"[clean_masks] Morphological closing "
                f"(radius={closing_radius_um} XY voxels, "
                f"iterations={closing_iterations}, "
                f"struct shape={struct.shape})..."
            )
            total_added = 0
            for i in range(closing_iterations):
                pre = int(np.sum(cleaned_mask))
                closed = binary_closing(cleaned_mask, structure=struct)
                # OR with original to preserve true signal at borders
                cleaned_mask = closed | mask.astype(bool)
                added = int(np.sum(cleaned_mask)) - pre
                total_added += added
                self.logger.info(
                    f"[clean_masks]  Closing pass {i + 1}/{closing_iterations}: "
                    f"+{added:,} voxels bridged"
                )
                if added == 0:
                    self.logger.info(
                        "[clean_masks]  No new voxels bridged -- stopping early"
                    )
                    stats['closing_iterations_run'] = i + 1
                    break
            else:
                stats['closing_iterations_run'] = closing_iterations
            stats['closing_voxels_added'] = total_added

        # ------------------------------------------------------------------
        # Step 2: Component filtering
        # ------------------------------------------------------------------
        self.logger.info("[clean_masks] Identifying connected components...")
        labeled_mask = label(cleaned_mask, connectivity=1)   # 6-connectivity
        num_components = int(labeled_mask.max())
        stats['original_components'] = num_components

        if num_components == 0:
            self.logger.warning("[clean_masks] Input mask is empty after closing")
            raise ValueError("Input mask is empty after closing -- no components found")

        component_sizes = np.bincount(labeled_mask.ravel())
        component_sizes[0] = 0   # exclude background

        self.logger.info(
            f"[clean_masks] Found {num_components} connected components -- "
            f"min={component_sizes[1:].min():,}  "
            f"max={component_sizes[1:].max():,}  "
            f"median={int(np.median(component_sizes[1:])):,}"
        )

        if keep_largest_only:
            largest_id = int(component_sizes.argmax())
            keep_ids = [largest_id]
            self.logger.info(
                f"[clean_masks] Keeping largest component "
                f"(ID {largest_id}, {component_sizes[largest_id]:,} voxels)"
            )
        else:
            keep_ids = list(np.where(component_sizes >= min_component_voxels)[0])
            self.logger.info(
                f"[clean_masks] Keeping {len(keep_ids)} components "
                f">= {min_component_voxels:,} voxels"
            )

        cleaned_mask = np.isin(labeled_mask, keep_ids).astype(bool)
        stats['kept_components'] = len(keep_ids)
        stats['removed_components'] = num_components - len(keep_ids)

        # ------------------------------------------------------------------
        # Step 3: Fill holes
        #
        # 3a. 3D fill -- scipy flood-fills from the exterior; seals voids that
        #     are fully enclosed in 3D after closing.
        #
        # 3b. Per-axis 2D fill (optional) -- fills holes in every cross-section
        #     slice along Z, Y, and X independently, then unions the results.
        #     This catches elongated voids (e.g. a gap running along the axis
        #     of a dendrite) that appear open-ended in 3D but are closed in
        #     every 2D cross-section.  Apply after the 3D fill so the 3D fill
        #     handles the easy cases first and per-axis fill only contributes
        #     the incremental difference.
        # ------------------------------------------------------------------
        if fill_holes:
            self.logger.info("[clean_masks] Filling internal holes and voids (3D)...")
            pre_fill = int(np.sum(cleaned_mask))
            filled = binary_fill_holes(cleaned_mask)
            assert filled is not None, "binary_fill_holes returned None"
            cleaned_mask = filled.astype(bool)
            post_fill = int(np.sum(cleaned_mask))
            added_3d = post_fill - pre_fill
            stats['holes_filled'] = True
            stats['hole_fill_voxels_added'] = added_3d
            if added_3d > 0:
                self.logger.info(f"[clean_masks]  3D fill: +{added_3d:,} voxels")
            else:
                self.logger.info("[clean_masks]  3D fill: no enclosed voids found")

            if fill_holes_per_axis:
                self.logger.info(
                    "[clean_masks] Per-axis 2D hole filling (Z, Y, X slices)..."
                )
                pre_axis = int(np.sum(cleaned_mask))
                cleaned_mask = _fill_holes_multiaxis(cleaned_mask)
                added_axis = int(np.sum(cleaned_mask)) - pre_axis
                stats['per_axis_fill_voxels_added'] = added_axis
                if added_axis > 0:
                    self.logger.info(
                        f"[clean_masks]  Per-axis fill caught additional "
                        f"+{added_axis:,} voxels not enclosed in 3D"
                    )
                else:
                    self.logger.info(
                        "[clean_masks]  Per-axis fill: no additional voids found"
                    )

        # ------------------------------------------------------------------
        # Step 4: Morphological smoothing (optional surface regularisation)
        # ------------------------------------------------------------------
        if smooth_iterations > 0:
            self.logger.info(f"[clean_masks] Smoothing ({smooth_iterations} iterations)...")
            for _ in range(smooth_iterations):
                eroded = binary_erosion(cleaned_mask)
                assert eroded is not None
                dilated = binary_dilation(eroded.astype(bool))
                assert dilated is not None
                cleaned_mask = dilated.astype(bool)

        # ------------------------------------------------------------------
        # Final stats
        # ------------------------------------------------------------------
        assert isinstance(cleaned_mask, np.ndarray), "cleaned_mask is not an ndarray"
        final_voxel_count = int(np.sum(cleaned_mask))
        stats['final_voxel_count'] = final_voxel_count
        stats['voxels_added'] = max(0, final_voxel_count - original_voxel_count)
        stats['voxels_removed'] = max(0, original_voxel_count - final_voxel_count)

        self.logger.info(
            f"[clean_masks] Cleaning complete: {original_voxel_count:,} -> {final_voxel_count:,} voxels "
            f"({100.0 * final_voxel_count / max(original_voxel_count, 1):.1f}% of input)"
        )

        return VoxelData(
            mask=cleaned_mask,
            bbox_min_vox=bbox_min_vox,
            voxel_size_um=voxel_size_um or (1.0, 1.0, 1.0),
        ), stats

    def get_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Return only the largest connected component."""
        labeled_mask = label(mask, connectivity=1)
        if labeled_mask.max() == 0:
            return mask.copy()
        component_sizes = np.bincount(labeled_mask.ravel())
        component_sizes[0] = 0
        return (labeled_mask == int(component_sizes.argmax())).astype(bool)


class MeshCleaner:
    """
    Clean surface meshes (vertices and faces).

    Operations:
    - Remove duplicate vertices
    - Remove degenerate faces
    - Remove isolated vertices
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def clean_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        remove_duplicates: bool = True,
        remove_degenerate: bool = True,
        remove_isolated: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Clean a surface mesh.

        Args:
            vertices          : (N, 3) vertex coordinates.
            faces             : (M, 3) triangle indices.
            remove_duplicates : Merge duplicate vertices and remap faces.
            remove_degenerate : Remove zero-area triangles.
            remove_isolated   : Remove vertices unreferenced by any face.

        Returns:
            (cleaned_vertices, cleaned_faces, stats)
        """
        self.logger.info("[clean_mesh] Starting mesh cleaning...")

        stats = {
            'original_vertices': len(vertices),
            'original_faces': len(faces),
            'duplicate_vertices_removed': 0,
            'degenerate_faces_removed': 0,
            'isolated_vertices_removed': 0,
            'final_vertices': 0,
            'final_faces': 0,
        }

        clean_verts = vertices.copy()
        clean_faces = faces.copy()

        if remove_degenerate:
            self.logger.info("[clean_mesh] Removing degenerate faces...")
            valid = (
                (clean_faces[:, 0] != clean_faces[:, 1]) &
                (clean_faces[:, 1] != clean_faces[:, 2]) &
                (clean_faces[:, 2] != clean_faces[:, 0])
            )
            n_removed = int(len(clean_faces) - np.sum(valid))
            clean_faces = clean_faces[valid]
            stats['degenerate_faces_removed'] = n_removed
            if n_removed:
                self.logger.info(f"  Removed {n_removed} degenerate faces")

        if remove_duplicates:
            self.logger.info("[clean_mesh] Removing duplicate vertices...")
            unique_verts, inv = np.unique(clean_verts, axis=0, return_inverse=True)
            n_removed = int(len(clean_verts) - len(unique_verts))
            if n_removed:
                clean_faces = inv[clean_faces]
                clean_verts = unique_verts
                stats['duplicate_vertices_removed'] = n_removed
                self.logger.info(f"  Removed {n_removed} duplicate vertices")

        if remove_isolated:
            self.logger.info("[clean_mesh] Removing isolated vertices...")
            used = np.unique(clean_faces.ravel())
            n_removed = int(len(clean_verts) - len(used))
            if n_removed:
                remap = np.full(len(clean_verts), -1, dtype=np.int64)
                remap[used] = np.arange(len(used))
                clean_faces = remap[clean_faces]
                clean_verts = clean_verts[used]
                stats['isolated_vertices_removed'] = n_removed
                self.logger.info(f"[clean_mesh]   Removed {n_removed} isolated vertices")

        stats['final_vertices'] = len(clean_verts)
        stats['final_faces'] = len(clean_faces)

        self.logger.info(
            f"[clean_mesh] Mesh cleaning complete: "
            f"verts {stats['original_vertices']:,} -> {stats['final_vertices']:,}  "
            f"faces {stats['original_faces']:,} -> {stats['final_faces']:,}"
        )

        return clean_verts, clean_faces, stats


def quick_clean_mask(
    mask: np.ndarray,
    keep_largest: bool = True,
    fill_holes: bool = True,
    closing_radius: int = 2,
    voxel_size_um: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    """One-call mask cleaning with gap-bridging closing."""
    cleaner = VoxelCleaner()
    cleaned, _ = cleaner.clean_mask(
        mask,
        keep_largest_only=keep_largest,
        fill_holes=fill_holes,
        closing_radius_um=closing_radius,
        voxel_size_um=voxel_size_um,
    )
    return cleaned.mask


def quick_clean_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """One-call mesh cleaning."""
    cleaner = MeshCleaner()
    clean_verts, clean_faces, _ = cleaner.clean_mesh(vertices, faces)
    return clean_verts, clean_faces