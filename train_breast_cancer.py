# ============================================================
#   Breast Cancer Detection - Anti-Overfitting Training
#   Dataset : 2271 train | 160 valid | 80 test
#   GPU     : NVIDIA GTX 1660 Ti (6GB VRAM)
#   Fixes   : Dropout, Augmentation, LR, Early Stop, Regularization
# ============================================================

import os
import torch
from roboflow import Roboflow
from ultralytics import YOLO


if __name__ == '__main__':

    # ── STEP 1: Check GPU ────────────────────────────────────
    print("=" * 55)
    print("   🖥️  SYSTEM CHECK")
    print("=" * 55)

    if torch.cuda.is_available():
        print(f"   ✅ GPU   : {torch.cuda.get_device_name(0)}")
        print(f"   ✅ VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        DEVICE = 0
    else:
        print("   ⚠️  No GPU — using CPU")
        DEVICE = "cpu"

    print("=" * 55)


    # ── STEP 2: Dataset ──────────────────────────────────────
    print("\n📥 Loading dataset...")

    rf      = Roboflow(api_key="pDYKSbs6z9GR4En2VnN4")
    project = rf.workspace("breast-cancer-4qfmz").project("cancer-detecion")
    version = project.version(1)
    dataset = version.download("yolov8")

    DATA_YAML = os.path.join(dataset.location, "data.yaml")
    print(f"✅ Dataset ready: {DATA_YAML}")


    # ── STEP 3: Load Model ───────────────────────────────────
    print("\n🤖 Loading YOLOv8s...")
    model = YOLO("yolov8s.pt")


    # ── STEP 4: Train with Overfitting Fixes ─────────────────
    print("\n" + "=" * 55)
    print("   🚀 ANTI-OVERFITTING TRAINING")
    print("=" * 55)
    print("   ✅ Fix 1 : Dropout = 0.3")
    print("   ✅ Fix 2 : Heavy Augmentation ON")
    print("   ✅ Fix 3 : Weight Decay increased")
    print("   ✅ Fix 4 : Early Stopping = 10 epochs")
    print("   ✅ Fix 5 : Lower Learning Rate")
    print("   ✅ Fix 6 : Mosaic + MixUp augmentation")
    print("   ✅ Fix 7 : Reduced Epochs to 40")
    print("=" * 55 + "\n")

    results = model.train(

        # ── Core Settings ─────────────────────────────────
        data          = DATA_YAML,
        epochs        = 40,             # ✅ reduced (was 50)
        imgsz         = 640,
        batch         = 16,
        device        = DEVICE,
        name          = "breast_cancer_v2_fixed",
        workers       = 0,              # Windows fix

        # ── Overfitting Fixes ─────────────────────────────
        dropout       = 0.3,            # ✅ FIX 1: was 0.0
        weight_decay  = 0.001,          # ✅ FIX 2: was 0.0005 (doubled)
        patience      = 10,             # ✅ FIX 3: early stop sooner

        # ── Learning Rate ─────────────────────────────────
        optimizer     = "Adam",
        lr0           = 0.0005,         # ✅ FIX 4: was 0.001 (halved)
        lrf           = 0.01,
        warmup_epochs = 3,
        cos_lr        = True,

        # ── Heavy Augmentation (prevents memorizing) ──────
        mosaic        = 1.0,            # ✅ FIX 5: mosaic augmentation
        mixup         = 0.2,            # ✅ FIX 6: mixup augmentation
        copy_paste    = 0.1,            # ✅ FIX 7: copy-paste augment
        degrees       = 15.0,           # rotation
        translate     = 0.1,            # translation
        scale         = 0.5,            # scaling
        shear         = 5.0,            # shear
        perspective   = 0.0001,         # perspective
        flipud        = 0.5,            # vertical flip
        fliplr        = 0.5,            # horizontal flip
        hsv_h         = 0.015,          # hue augment
        hsv_s         = 0.7,            # saturation augment
        hsv_v         = 0.4,            # brightness augment

        # ── Other ─────────────────────────────────────────
        pretrained    = True,
        amp           = False,          # GTX 1660 Ti fix
        plots         = True,
        save          = True,
        save_period   = 5,
        verbose       = True
    )

    print("\n✅ Training Complete!")


    # ── STEP 5: Validate ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("   📊 VALIDATION RESULTS")
    print("=" * 55)

    metrics = model.val()

    map50    = metrics.box.map50
    map5095  = metrics.box.map
    precision= metrics.box.mp
    recall   = metrics.box.mr

    print(f"   mAP@50       : {map50:.4f}")
    print(f"   mAP@50-95    : {map5095:.4f}")
    print(f"   Precision    : {precision:.4f}")
    print(f"   Recall       : {recall:.4f}")

    # ── Overfitting Check ─────────────────────────────────
    print("\n" + "=" * 55)
    print("   🔍 OVERFITTING CHECK")
    print("=" * 55)

    if map50 >= 0.90:
        print("   ✅ mAP@50 is strong — model generalizes well!")
    elif map50 >= 0.75:
        print("   ⚠️  mAP@50 is decent — slight underfitting possible")
    else:
        print("   ❌ mAP@50 dropped too much — try dropout = 0.1")

    print("=" * 55)


    # ── STEP 6: Test Predictions ─────────────────────────────
    print("\n🔍 Running on test set (80 images)...")

    best_model = YOLO("runs/detect/breast_cancer_v2_fixed/weights/best.pt")
    test_path  = os.path.join(dataset.location, "test", "images")

    best_model.predict(
        source    = test_path,
        conf      = 0.25,
        iou       = 0.45,
        save      = True,
        save_txt  = True,
        name      = "breast_cancer_v2_test"
    )

    print("✅ Test predictions saved!")


    # ── STEP 7: Summary ──────────────────────────────────────
    print("\n" + "=" * 55)
    print("   🎉 DONE!")
    print("=" * 55)
    print(f"   Best Model : runs/detect/breast_cancer_v2_fixed/weights/best.pt")
    print(f"   mAP@50     : {map50:.4f}")
    print(f"   Precision  : {precision:.4f}")
    print(f"   Recall     : {recall:.4f}")
    print("=" * 55)