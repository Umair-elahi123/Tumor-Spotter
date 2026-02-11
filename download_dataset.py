# ============================================================
#   Breast Cancer Dataset - Download Only
#   Roboflow: cancer-detecion v1
# ============================================================

# ── STEP 1: Install Roboflow (run in terminal first) ────────
# pip install roboflow

from roboflow import Roboflow
import os

# ── STEP 2: Download Dataset ─────────────────────────────────
print("📥 Connecting to Roboflow...")

rf = Roboflow(api_key="pDYKSbs6z9GR4En2VnN4")
project = rf.workspace("breast-cancer-4qfmz").project("cancer-detecion")
version = project.version(1)

print("📦 Downloading dataset in YOLOv8 format...")
dataset = version.download("yolov8")

print(f"\n✅ Dataset downloaded successfully!")
print(f"📁 Location: {dataset.location}")

# ── STEP 3: Show data.yaml contents ──────────────────────────
yaml_path = os.path.join(dataset.location, "data.yaml")

print("\n" + "="*50)
print("📄 Your data.yaml file content:")
print("="*50)

with open(yaml_path, "r") as f:
    content = f.read()
    print(content)

print("="*50)
print(f"✅ data.yaml found at: {yaml_path}")
print("="*50)