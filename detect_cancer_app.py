# ============================================================
#   Breast Cancer Detection App
#   Upload image → Detect cancer → Show results
#   GUI: Gradio (runs in browser automatically)
# ============================================================

# ── INSTALL FIRST IN TERMINAL ────────────────────────────────
# pip install gradio ultralytics opencv-python pillow

import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os


# ── LOAD TRAINED MODEL ───────────────────────────────────────
MODEL_PATH = "runs/detect/breast_cancer_v2_fixed/weights/epoch35.pt"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at: {MODEL_PATH}")
    print("   Make sure you have trained the model first!")
    exit()

print("🤖 Loading breast cancer detection model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!")


# ── DETECTION FUNCTION ───────────────────────────────────────
def detect_cancer(image, confidence_threshold=0.1):
    """
    Takes uploaded image, runs YOLOv8 detection,
    returns annotated image + result summary
    """

    if image is None:
        return None, "⚠️ Please upload an image first."

    # Convert PIL image to numpy array (OpenCV format)
    img_array = np.array(image)

    # Run detection
    results = model.predict(
        source    = img_array,
        conf      = confidence_threshold,
        iou       = 0.45,
        verbose   = False
    )

    result      = results[0]
    detections  = result.boxes

    # Draw results on image
    annotated   = result.plot()

    # Convert BGR → RGB for display
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    output_image  = Image.fromarray(annotated_rgb)

    # ── Build Result Summary ──────────────────────────────────
    num_detections = len(detections)

    if num_detections == 0:
        status  = "✅ NO CANCER DETECTED"
        summary = f"""
<div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border: 2px solid #10b981; border-radius: 12px; padding: 25px; font-family: 'Segoe UI', Arial, sans-serif;">
    <div style="text-align: center; border-bottom: 2px solid #059669; padding-bottom: 15px; margin-bottom: 20px;">
        <h2 style="color: #065f46; margin: 0; font-size: 1.5em;">🔬 BREAST CANCER DETECTION</h2>
    </div>
    
    <div style="background: white; border-radius: 8px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #d1fae5;">
                <td style="padding: 10px; font-weight: bold; color: #047857;">Status:</td>
                <td style="padding: 10px; color: #065f46;">✅ No Cancer Detected</td>
            </tr>
            <tr style="border-bottom: 1px solid #d1fae5;">
                <td style="padding: 10px; font-weight: bold; color: #047857;">Detections:</td>
                <td style="padding: 10px; color: #065f46;">0</td>
            </tr>
            <tr>
                <td style="padding: 10px; font-weight: bold; color: #047857;">Confidence:</td>
                <td style="padding: 10px; color: #065f46;">{confidence_threshold:.0%} threshold</td>
            </tr>
        </table>
    </div>
    
    <div style="background: #a7f3d0; border-left: 4px solid #059669; border-radius: 6px; padding: 15px; margin-top: 15px;">
        <p style="margin: 0; color: #065f46; font-size: 0.95em;">
            <strong>ℹ️ Information:</strong> No suspicious regions found. Always consult a medical professional for proper diagnosis.
        </p>
    </div>
</div>
"""
    else:
        status  = f"⚠️ {num_detections} SUSPICIOUS REGION(S) DETECTED"

        detection_details = ""
        for i, box in enumerate(detections):
            conf       = float(box.conf[0])
            cls_id     = int(box.cls[0])
            cls_name   = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width      = x2 - x1
            height     = y2 - y1

            detection_details += f"""
            <div style="background: white; border-left: 4px solid #f59e0b; border-radius: 6px; padding: 15px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <h4 style="margin: 0 0 10px 0; color: #92400e;">📍 Detection #{i+1}</h4>
                <table style="width: 100%; font-size: 0.9em;">
                    <tr>
                        <td style="padding: 5px; color: #78350f; font-weight: bold;">Class:</td>
                        <td style="padding: 5px; color: #92400e;">{cls_name}</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; color: #78350f; font-weight: bold;">Confidence:</td>
                        <td style="padding: 5px; color: #92400e;">{conf:.1%}</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; color: #78350f; font-weight: bold;">Location:</td>
                        <td style="padding: 5px; color: #92400e;">({x1}, {y1}) → ({x2}, {y2})</td>
                    </tr>
                    <tr>
                        <td style="padding: 5px; color: #78350f; font-weight: bold;">Size:</td>
                        <td style="padding: 5px; color: #92400e;">{width}×{height} px</td>
                    </tr>
                </table>
            </div>
"""

        summary = f"""
<div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 2px solid #f59e0b; border-radius: 12px; padding: 25px; font-family: 'Segoe UI', Arial, sans-serif;">
    <div style="text-align: center; border-bottom: 2px solid #d97706; padding-bottom: 15px; margin-bottom: 20px;">
        <h2 style="color: #92400e; margin: 0; font-size: 1.5em;">🔬 BREAST CANCER DETECTION</h2>
    </div>
    
    <div style="background: white; border-radius: 8px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #fde68a;">
                <td style="padding: 10px; font-weight: bold; color: #b45309;">Status:</td>
                <td style="padding: 10px; color: #92400e;">⚠️ Cancer Detected</td>
            </tr>
            <tr style="border-bottom: 1px solid #fde68a;">
                <td style="padding: 10px; font-weight: bold; color: #b45309;">Detections:</td>
                <td style="padding: 10px; color: #92400e;">{num_detections} region(s)</td>
            </tr>
            <tr>
                <td style="padding: 10px; font-weight: bold; color: #b45309;">Confidence:</td>
                <td style="padding: 10px; color: #92400e;">{confidence_threshold:.0%} threshold</td>
            </tr>
        </table>
    </div>
    
    <div style="margin: 20px 0;">
        <h3 style="color: #92400e; margin-bottom: 15px; font-size: 1.1em;">📍 DETECTION DETAILS:</h3>
        {detection_details}
    </div>
    
    <div style="background: #fca5a5; border-left: 4px solid #dc2626; border-radius: 6px; padding: 15px; margin-top: 15px;">
        <p style="margin: 0; color: #7f1d1d; font-size: 0.95em;">
            </strong> Please consult a medical professional immediately for proper diagnosis and treatment.
        </p>
    </div>
</div>
"""

    return output_image, summary


# ── BUILD GRADIO UI ──────────────────────────────────────────
with gr.Blocks(
    title="Breast Cancer Detection",
    theme=gr.themes.Soft(
        primary_hue="pink",
        secondary_hue="rose",
    ),
    css="""
        .title-text { text-align: center; color: #be185d; }
        .subtitle   { text-align: center; color: #6b7280; margin-bottom: 20px; }
        .result-box { font-family: monospace; font-size: 14px; }
        footer      { display: none !important; }
    """
) as app:

    # ── Header ───────────────────────────────────────────────
    gr.HTML("""
        <div style="text-align:center; padding: 20px 0 10px 0;">
            <h1 style="color:#be185d; font-size:2.2em; margin:0;">
                TumorSpotter 
            </h1>
            <p style="color:#6b7280; font-size:1em; margin-top:8px;">
                Upload a mammogram image to detect suspicious regions
            </p>
            <p style="color:#9ca3af; font-size:0.85em;">
                Powered by YOLOv8s · Trained on 2271 images · mAP@50: 0.97 · mAP@50-95: 0.76
            </p>
        </div>
    """)

    gr.Markdown("---")

    # ── Main Layout ──────────────────────────────────────────
    with gr.Row():

        # Left Column - Input
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Image")

            input_image = gr.Image(
                type            = "pil",
                label           = "Upload Mammogram",
                height          = 350,
            )

            detect_btn = gr.Button(
                "🔍 Detect Cancer",
                variant = "primary",
                size    = "lg"
            )

            gr.Markdown("""
            > ⚕️ **Disclaimer:** This tool is for research purposes only.
            > Always consult a qualified medical professional for diagnosis.
            """)

        # Right Column - Output
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Detection Results")

            output_image = gr.Image(
                label  = "Annotated Image",
                height = 350,
            )

            output_text = gr.HTML(
                label    = "📋 Result Summary",
            )

    # ── Examples ─────────────────────────────────────────────
    gr.Markdown("---")
    gr.HTML("""
        <div style="margin: 20px 0;">
            <h3 style="color: #be185d; font-size: 1.3em; margin-bottom: 15px;">💡 How to Use</h3>
            <div style="display:flex; gap:20px; flex-wrap:wrap; padding: 10px 0;">
                <div style="background:#be185d; border:2px solid #fbcfe8; border-radius:8px; padding:20px; flex:1; min-width:250px;">
                    <div style="font-size: 1.5em; margin-bottom: 10px;">📤</div>
                    <b style="color: #f2f7f7; font-size: 1.1em;">Step 1</b><br>
                    <span style="color: #f2f7f7;">Upload your breast imaging scan</span>
                </div>
                <div style="background:#be185d; border:2px solid #fbcfe8; border-radius:8px; padding:20px; flex:1; min-width:250px;">
                    <div style="font-size: 1.5em; margin-bottom: 10px;">🔍</div>
                    <b style="color: #f2f7f7; font-size: 1.1em;">Step 2</b><br>
                    <span style="color: #f2f7f7;">Click "Detect Cancer" button</span>
                </div>
                <div style="background:#be185d; border:2px solid #fbcfe8; border-radius:8px; padding:20px; flex:1; min-width:250px;">
                    <div style="font-size: 1.5em; margin-bottom: 10px;">📊</div>
                    <b style="color: #f2f7f7; font-size: 1.1em;">Step 3</b><br>
                    <span style="color: #f2f7f7;">View annotated image and results</span>
                </div>
            </div>
        </div>
    """)

    # ── Button Action ─────────────────────────────────────────
    detect_btn.click(
        fn      = detect_cancer,
        inputs  = [input_image],
        outputs = [output_image, output_text]
    )

    # Also detect on image upload
    input_image.change(
        fn      = detect_cancer,
        inputs  = [input_image],
        outputs = [output_image, output_text]
    )


# ── LAUNCH APP ───────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  🚀 Launching Breast Cancer Detection App")
    print("=" * 50)
    print("  Opening browser automatically...")
    print("  URL: http://localhost:7860")
    print("=" * 50 + "\n")

    app.launch(
        inbrowser = True,    # auto open browser
        share     = False,   # set True for public link
        server_port = 7860
    )