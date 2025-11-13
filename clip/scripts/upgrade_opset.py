#!/usr/bin/env python3

import sys
import onnx
from onnx import version_converter, shape_inference
import onnx_graphsurgeon as gs


def upgrade_opset(input_path: str, output_path: str, target_opset: int = 16):
    print(f"\n🔄 Upgrading model: {input_path}")

    model = onnx.load(input_path)

    try:
        model = shape_inference.infer_shapes(model)
        print("Shape inference completed.")
    except Exception as e:
        print(f"Shape inference failed: {e}")

    # Convert to desired opset
    current_opset = model.opset_import[0].version
    print(f"Current opset: {current_opset} -> Target opset: {target_opset}")
    model = version_converter.convert_version(model, target_opset)

    # Clean up and sort graph using GraphSurgeon
    graph = gs.import_onnx(model)
    graph.cleanup().toposort()
    model = gs.export_onnx(graph)

    onnx.save(model, output_path)
    print(f"Saved upgraded model to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python upgrade_opset.py <input_model.onnx> <output_model.onnx>")
        sys.exit(1)

    input_model = sys.argv[1]
    output_model = sys.argv[2]
    opset = int(sys.argv[3]) if len(sys.argv) > 3 else 16

    upgrade_opset(input_model, output_model, opset)
