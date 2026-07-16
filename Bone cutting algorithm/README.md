# Bone cutting path planner

This workspace now contains a desktop-style workflow for 3D point-cloud path planning:

- First, select a PLY point-cloud model from disk.
- Then click two points in the 3D window to choose the start and end points.
- The program computes a path using A* on a K-nearest-neighbor graph.
- It then smooths the path with cubic spline interpolation and saves the result.

## Files

- `path_planner.py`: core path-planning and spline smoothing logic
- `demo.py`: desktop GUI entry point for selecting a point cloud and two endpoints

## Run the interactive program

```bash
python demo.py
```

If you want to skip the interactive GUI and run on a specific PLY file:

```bash
python demo.py --ply "C:/path/to/your/model.ply"
```

The results are written to the `outputs/` folder:

- `outputs/path_discrete.csv`
- `outputs/path_smooth.csv`
- `outputs/selection.csv`
