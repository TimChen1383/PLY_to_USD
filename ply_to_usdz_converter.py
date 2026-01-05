#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone PLY to USDZ converter for 3D Gaussian Splatting models.
# No CUDA or GPU required - works on any Windows/Mac/Linux machine.
#
# Requirements:
#   pip install plyfile usd-core msgpack numpy
#
# Usage:
#   python ply_to_usdz_standalone.py input.ply -o output.usdz
#   python ply_to_usdz_standalone.py input.ply  # outputs input.usdz

import argparse
import gzip
import io
import logging
import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import msgpack
import numpy as np
from plyfile import PlyData

# USD imports
from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils, UsdVol

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class NamedSerialized:
    """Stores serialized data with a filename."""

    filename: str
    serialized: Union[str, bytes]

    def save_to_zip(self, zip_file: zipfile.ZipFile):
        zip_file.writestr(self.filename, self.serialized)


@dataclass
class NamedUSDStage:
    """Stores a USD stage with a filename."""

    filename: str
    stage: Usd.Stage

    def save_to_zip(self, zip_file: zipfile.ZipFile):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=self.filename, delete=False) as temp_file:
            temp_file_path = temp_file.name
        self.stage.GetRootLayer().Export(temp_file_path)
        with open(temp_file_path, "rb") as file:
            usd_data = file.read()
        zip_file.writestr(self.filename, usd_data)
        os.unlink(temp_file_path)


# ==============================================================================
# PLY Reading
# ==============================================================================


def read_ply_gaussians(ply_path: str, max_sh_degree: int = 3) -> Dict[str, np.ndarray]:
    """
    Read 3D Gaussian Splatting data from a PLY file.

    Args:
        ply_path: Path to the PLY file
        max_sh_degree: Maximum spherical harmonics degree (default: 3)

    Returns:
        Dictionary containing positions, rotations, scales, densities,
        features_albedo, features_specular, and n_active_features
    """
    logger.info(f"Reading PLY file: {ply_path}")
    plydata = PlyData.read(ply_path)

    # Positions
    positions = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    ).astype(np.float32)

    num_gaussians = positions.shape[0]
    logger.info(f"Found {num_gaussians} Gaussians")

    # Densities (opacity)
    densities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

    # Albedo (DC components of spherical harmonics)
    features_albedo = np.zeros((num_gaussians, 3), dtype=np.float32)
    features_albedo[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_albedo[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_albedo[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

    # Specular (higher-order spherical harmonics)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    num_speculars = (max_sh_degree + 1) ** 2 - 1
    expected_extra_f_count = 3 * num_speculars

    features_specular = np.zeros((num_gaussians, num_speculars * 3), dtype=np.float32)
    if len(extra_f_names) == expected_extra_f_count:
        temp_specular = np.zeros((num_gaussians, expected_extra_f_count), dtype=np.float32)
        for idx, attr_name in enumerate(extra_f_names):
            temp_specular[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape from (N, 3*num_speculars) to (N, 3, num_speculars) then transpose
        temp_specular = temp_specular.reshape((num_gaussians, 3, num_speculars))
        features_specular = temp_specular.transpose(0, 2, 1).reshape((num_gaussians, num_speculars * 3))
    elif len(extra_f_names) == 0:
        logger.info("PLY file only contains DC components, initializing higher-order SH to zero")
    else:
        raise ValueError(
            f"Unexpected number of f_rest_ properties: found {len(extra_f_names)}, expected {expected_extra_f_count} or 0"
        )

    # Scales
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((num_gaussians, len(scale_names)), dtype=np.float32)
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # Rotations
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rotations = np.zeros((num_gaussians, len(rot_names)), dtype=np.float32)
    for idx, attr_name in enumerate(rot_names):
        rotations[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return {
        "positions": positions,
        "rotations": rotations,
        "scales": scales,
        "densities": densities,
        "features_albedo": features_albedo,
        "features_specular": features_specular,
        "n_active_features": max_sh_degree,
    }


# ==============================================================================
# NuRec Template Creation
# ==============================================================================


def create_nurec_template(
    positions: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    densities: np.ndarray,
    features_albedo: np.ndarray,
    features_specular: np.ndarray,
    n_active_features: int,
    density_activation: str = "sigmoid",
    scale_activation: str = "exp",
) -> Dict[str, Any]:
    """
    Create the NuRec template dictionary for USDZ export.

    This creates a format compatible with NVIDIA Omniverse Kit and Isaac Sim.
    """
    template = {
        "nre_data": {
            "version": "0.2.576",
            "model": "nre",
            "config": {
                "layers": {
                    "gaussians": {
                        "name": "sh-gaussians",
                        "device": "cuda",
                        "density_activation": density_activation,
                        "scale_activation": scale_activation,
                        "rotation_activation": "normalize",
                        "precision": 16,
                        "particle": {
                            "density_kernel_planar": False,
                            "density_kernel_degree": 2,
                            "density_kernel_density_clamping": False,
                            "density_kernel_min_response": 0.0113,
                            "radiance_sph_degree": 3,
                        },
                        "transmittance_threshold": 0.001,
                    }
                },
                "renderer": {
                    "name": "3dgut-nrend",
                    "log_level": 3,
                    "force_update": False,
                    "update_step_train_batch_end": False,
                    "per_ray_features": False,
                    "global_z_order": False,
                    "projection": {
                        "n_rolling_shutter_iterations": 5,
                        "ut_dim": 3,
                        "ut_alpha": 1.0,
                        "ut_beta": 2.0,
                        "ut_kappa": 0.0,
                        "ut_require_all_sigma_points": False,
                        "image_margin_factor": 0.1,
                        "min_projected_ray_radius": 0.5477225575051661,
                    },
                    "culling": {
                        "rect_bounding": True,
                        "tight_opacity_bounding": True,
                        "tile_based": True,
                        "near_clip_distance": 0.2,
                        "far_clip_distance": 3.402823466e38,
                    },
                    "render": {"mode": "kbuffer", "k_buffer_size": 0},
                },
                "name": "gaussians_primitive",
                "appearance_embedding": {"name": "skip-appearance", "embedding_dim": 0, "device": "cuda"},
                "background": {"name": "skip-background", "device": "cuda", "composite_in_linear_space": False},
            },
            "state_dict": {
                "._extra_state": {"obj_track_ids": {"gaussians": []}},
            },
        }
    }

    # Fill state dict with tensor data (converted to float16 for efficiency)
    dtype = np.float16
    state_dict = template["nre_data"]["state_dict"]

    state_dict[".gaussians_nodes.gaussians.positions"] = positions.astype(dtype).tobytes()
    state_dict[".gaussians_nodes.gaussians.rotations"] = rotations.astype(dtype).tobytes()
    state_dict[".gaussians_nodes.gaussians.scales"] = scales.astype(dtype).tobytes()
    state_dict[".gaussians_nodes.gaussians.densities"] = densities.astype(dtype).tobytes()
    state_dict[".gaussians_nodes.gaussians.features_albedo"] = features_albedo.astype(dtype).tobytes()
    state_dict[".gaussians_nodes.gaussians.features_specular"] = features_specular.astype(dtype).tobytes()

    # Empty extra_signal tensor
    extra_signal = np.zeros((positions.shape[0], 0), dtype=dtype)
    state_dict[".gaussians_nodes.gaussians.extra_signal"] = extra_signal.tobytes()

    # n_active_features as 64-bit integer
    state_dict[".gaussians_nodes.gaussians.n_active_features"] = np.array([n_active_features], dtype=np.int64).tobytes()

    # Store shapes
    state_dict[".gaussians_nodes.gaussians.positions.shape"] = list(positions.shape)
    state_dict[".gaussians_nodes.gaussians.rotations.shape"] = list(rotations.shape)
    state_dict[".gaussians_nodes.gaussians.scales.shape"] = list(scales.shape)
    state_dict[".gaussians_nodes.gaussians.densities.shape"] = list(densities.shape)
    state_dict[".gaussians_nodes.gaussians.features_albedo.shape"] = list(features_albedo.shape)
    state_dict[".gaussians_nodes.gaussians.features_specular.shape"] = list(features_specular.shape)
    state_dict[".gaussians_nodes.gaussians.extra_signal.shape"] = list(extra_signal.shape)
    state_dict[".gaussians_nodes.gaussians.n_active_features.shape"] = []

    return template


# ==============================================================================
# USD/USDZ Export
# ==============================================================================


def initialize_usd_stage() -> Usd.Stage:
    """Initialize a new USD stage with standard settings."""
    stage = Usd.Stage.CreateInMemory()
    stage.SetMetadata("metersPerUnit", 1)
    stage.SetMetadata("upAxis", "Z")

    world_path = "/World"
    UsdGeom.Xform.Define(stage, world_path)
    stage.SetMetadata("defaultPrim", world_path[1:])

    return stage


def create_gauss_usd(model_filename: str, positions: np.ndarray) -> NamedUSDStage:
    """Create the USD stage containing the Gaussian volume."""
    logger.info("Creating USD stage for Gaussian volume")

    # Calculate bounding box
    min_coord = np.min(positions, axis=0)
    max_coord = np.max(positions, axis=0)
    logger.info(f"Bounding box: min={min_coord}, max={max_coord}")

    min_x, min_y, min_z = float(min_coord[0]), float(min_coord[1]), float(min_coord[2])
    max_x, max_y, max_z = float(max_coord[0]), float(max_coord[1]), float(max_coord[2])

    stage = initialize_usd_stage()

    # Render settings for Omniverse
    render_settings = {
        "rtx:rendermode": "RaytracedLighting",
        "rtx:directLighting:sampledLighting:samplesPerPixel": 8,
        "rtx:post:histogram:enabled": False,
        "rtx:post:registeredCompositing:invertToneMap": True,
        "rtx:post:registeredCompositing:invertColorCorrection": True,
        "rtx:material:enableRefraction": False,
        "rtx:post:tonemap:op": 2,
        "rtx:raytracing:fractionalCutoutOpacity": False,
        "rtx:matteObject:visibility:secondaryRays": True,
    }
    stage.SetMetadataByDictKey("customLayerData", "renderSettings", render_settings)

    # Define UsdVol::Volume
    gauss_path = "/World/gauss"
    gauss_volume = UsdVol.Volume.Define(stage, gauss_path)
    gauss_prim = gauss_volume.GetPrim()

    # Default conversion matrix from 3DGRUT to USDZ coordinate system
    default_conv_tf = np.array(
        [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )

    matrix_op = gauss_volume.AddTransformOp()
    matrix_op.Set(Gf.Matrix4d(*default_conv_tf.flatten()))

    # NuRec volume properties
    gauss_prim.CreateAttribute("omni:nurec:isNuRecVolume", Sdf.ValueTypeNames.Bool).Set(True)
    gauss_prim.CreateAttribute("omni:nurec:useProxyTransform", Sdf.ValueTypeNames.Bool).Set(False)

    # Define field assets
    density_field_path = gauss_path + "/density_field"
    density_field = stage.DefinePrim(density_field_path, "OmniNuRecFieldAsset")
    gauss_volume.CreateFieldRelationship("density", density_field_path)

    emissive_color_field_path = gauss_path + "/emissive_color_field"
    emissive_color_field = stage.DefinePrim(emissive_color_field_path, "OmniNuRecFieldAsset")
    gauss_volume.CreateFieldRelationship("emissiveColor", emissive_color_field_path)

    # Set file paths for field assets
    nurec_relative_path = "./" + model_filename
    density_field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set(nurec_relative_path)
    density_field.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set("density")
    density_field.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set("float")
    density_field.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set("density")

    emissive_color_field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set(nurec_relative_path)
    emissive_color_field.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set("emissiveColor")
    emissive_color_field.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set("float3")
    emissive_color_field.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set("emissiveColor")

    # Color correction matrix (identity)
    emissive_color_field.CreateAttribute("omni:nurec:ccmR", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f([1, 0, 0, 0]))
    emissive_color_field.CreateAttribute("omni:nurec:ccmG", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f([0, 1, 0, 0]))
    emissive_color_field.CreateAttribute("omni:nurec:ccmB", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f([0, 0, 1, 0]))

    # Extent and crop boundaries
    gauss_prim.GetAttribute("extent").Set([[min_x, min_y, min_z], [max_x, max_y, max_z]])
    gauss_prim.CreateAttribute("omni:nurec:offset", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3d(0, 0, 0))
    gauss_prim.CreateAttribute("omni:nurec:crop:minBounds", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3d(min_x, min_y, min_z))
    gauss_prim.CreateAttribute("omni:nurec:crop:maxBounds", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3d(max_x, max_y, max_z))

    # Empty proxy mesh relationship
    gauss_prim.CreateRelationship("proxy")

    return NamedUSDStage(filename="gauss.usda", stage=stage)


def create_default_usd(gauss_stage: NamedUSDStage) -> NamedUSDStage:
    """Create the default USD layer that references the gauss stage."""
    stage = initialize_usd_stage()

    # Silence dangling reference warnings
    delegate = UsdUtils.CoalescingDiagnosticDelegate()

    prim = stage.OverridePrim(f"/World/{Path(gauss_stage.filename).stem}")
    prim.GetReferences().AddReference(gauss_stage.filename)

    # Copy render settings
    gauss_layer = gauss_stage.stage.GetRootLayer()
    if "renderSettings" in gauss_layer.customLayerData:
        new_settings = gauss_layer.customLayerData["renderSettings"]
        stage.SetMetadataByDictKey("customLayerData", "renderSettings", new_settings)

    return NamedUSDStage(filename="default.usda", stage=stage)


def write_usdz(
    output_path: Path,
    model_file: NamedSerialized,
    gauss_usd: NamedUSDStage,
    default_usd: NamedUSDStage,
) -> None:
    """Write the final USDZ file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zip_file:
        # default.usda must be first (USDZ spec)
        default_usd.save_to_zip(zip_file)
        model_file.save_to_zip(zip_file)
        gauss_usd.save_to_zip(zip_file)

    logger.info(f"USDZ file created: {output_path}")


# ==============================================================================
# Main Conversion Function
# ==============================================================================


def convert_ply_to_usdz(input_path: Path, output_path: Path, max_sh_degree: int = 3) -> None:
    """
    Convert a 3D Gaussian Splatting PLY file to USDZ format.

    Args:
        input_path: Path to input PLY file
        output_path: Path to output USDZ file
        max_sh_degree: Maximum spherical harmonics degree (default: 3)
    """
    # Read PLY
    gaussian_data = read_ply_gaussians(str(input_path), max_sh_degree)

    # Create NuRec template
    template = create_nurec_template(
        positions=gaussian_data["positions"],
        rotations=gaussian_data["rotations"],
        scales=gaussian_data["scales"],
        densities=gaussian_data["densities"],
        features_albedo=gaussian_data["features_albedo"],
        features_specular=gaussian_data["features_specular"],
        n_active_features=gaussian_data["n_active_features"],
    )

    # Compress template data
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=0) as f:
        packed = msgpack.packb(template)
        f.write(packed)

    model_file = NamedSerialized(filename=output_path.stem + ".nurec", serialized=buffer.getvalue())

    # Create USD stages
    gauss_usd = create_gauss_usd(model_file.filename, gaussian_data["positions"])
    default_usd = create_default_usd(gauss_usd)

    # Write USDZ
    write_usdz(output_path, model_file, gauss_usd, default_usd)


# ==============================================================================
# CLI
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3D Gaussian Splatting PLY files to USDZ format for NVIDIA Omniverse.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ply_to_usdz_standalone.py model.ply
  python ply_to_usdz_standalone.py model.ply -o output.usdz
  python ply_to_usdz_standalone.py model.ply --sh-degree 2

Requirements:
  pip install plyfile usd-core msgpack numpy
        """,
    )
    parser.add_argument("input", type=str, help="Input PLY file path")
    parser.add_argument("-o", "--output", type=str, help="Output USDZ file path (default: input with .usdz extension)")
    parser.add_argument("--sh-degree", type=int, default=3, help="Max spherical harmonics degree (default: 3)")

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    if input_path.suffix.lower() != ".ply":
        logger.error(f"Input must be a PLY file: {input_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_suffix(".usdz")

    logger.info(f"Converting: {input_path} -> {output_path}")
    convert_ply_to_usdz(input_path, output_path, args.sh_degree)
    logger.info("Done!")


if __name__ == "__main__":
    main()
