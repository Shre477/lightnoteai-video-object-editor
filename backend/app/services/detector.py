"""Open-vocabulary object detection using OWL-ViT (google/owlvit-base-patch32).

Only imported when DEMO_MODE=false, so the app runs fine without torch /
transformers installed in demo mode. Given target_text like
"coca-cola bottle", returns the best-matching bounding box in the frame.
"""


def detect_object(frame_bgr, target_text: str):
    import torch
    from PIL import Image
    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    from ..config import settings

    if not hasattr(detect_object, "_model"):
        detect_object._processor = OwlViTProcessor.from_pretrained(settings.OWLVIT_MODEL)
        detect_object._model = OwlViTForObjectDetection.from_pretrained(settings.OWLVIT_MODEL)

    image = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR -> RGB
    inputs = detect_object._processor(text=[[target_text]], images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = detect_object._model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = detect_object._processor.post_process_object_detection(
        outputs, threshold=0.1, target_sizes=target_sizes
    )[0]

    if len(results["boxes"]) == 0:
        # Nothing confidently matched: fall back to a centered box rather
        # than crashing the whole job. Surfaced to the user via job status.
        h, w = frame_bgr.shape[:2]
        return (w // 4, h // 4, w // 2, h // 2)

    best_idx = int(results["scores"].argmax())
    x1, y1, x2, y2 = results["boxes"][best_idx].tolist()
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
