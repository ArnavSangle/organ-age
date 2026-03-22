"""Check which packages are available."""
packages = ["plotly", "kaleido", "pyvista", "nibabel", "umap", "scipy", "PIL"]
for pkg in packages:
    try:
        if pkg == "umap":
            import umap
        elif pkg == "PIL":
            from PIL import Image
        else:
            __import__(pkg)
        print(f"  OK: {pkg}")
    except ImportError as e:
        print(f"  MISSING: {pkg} ({e})")
