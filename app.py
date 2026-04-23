import io
import time
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.request import urlopen

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50

st.set_page_config(
    page_title="Image Processor - FCN / R-CNN Family / Mask R-CNN",
    page_icon="🧠",
    layout="wide",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REFERENCE_METRICS = [
    {
        "Method": "R-CNN (2014)",
        "Reference": "VOC07 mAP",
        "Value": "~58.5",
        "Approx FPS": "~0.05",
    },
    {
        "Method": "Fast R-CNN (2015)",
        "Reference": "VOC07 mAP",
        "Value": "~66.9",
        "Approx FPS": "~0.5",
    },
    {
        "Method": "Faster R-CNN (2015)",
        "Reference": "VOC07 mAP",
        "Value": "~73.2",
        "Approx FPS": "~5",
    },
    {
        "Method": "Mask R-CNN (2017)",
        "Reference": "COCO Mask AP",
        "Value": "~35.7",
        "Approx FPS": "~5",
    },
    {
        "Method": "FCN (2015)",
        "Reference": "PASCAL mIoU",
        "Value": "~62.2",
        "Approx FPS": "n/a",
    },
]


@st.cache_resource(show_spinner=False)
def load_fcn_model():
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights).to(DEVICE)
    model.eval()
    return model, weights


@st.cache_resource(show_spinner=False)
def load_faster_rcnn_model():
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to(DEVICE)
    model.eval()
    categories = weights.meta.get("categories", [])
    return model, weights, categories


@st.cache_resource(show_spinner=False)
def load_fast_proxy_model():
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(DEVICE)
    model.eval()
    categories = weights.meta.get("categories", [])
    return model, weights, categories


@st.cache_resource(show_spinner=False)
def load_mask_rcnn_model():
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights).to(DEVICE)
    model.eval()
    categories = weights.meta.get("categories", [])
    return model, weights, categories


@st.cache_resource(show_spinner=False)
def load_region_classifier():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(DEVICE)
    model.eval()
    return model, weights


def color_for_index(index: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(index + 17)
    color = rng.integers(32, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def safe_category_name(label_idx: int, categories: Sequence[str]) -> str:
    if 0 <= label_idx < len(categories):
        return str(categories[label_idx])
    return f"class_{label_idx}"


def load_uploaded_or_url_image() -> Optional[Image.Image]:
    uploaded = st.sidebar.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        help="For best visual quality, use a photo with objects and scenes.",
    )

    url_value = st.sidebar.text_input(
        "Or load image from URL",
        value="",
        placeholder="https://example.com/demo.jpg",
    )

    if uploaded is not None:
        return Image.open(uploaded).convert("RGB")

    if url_value.strip():
        try:
            with urlopen(url_value.strip(), timeout=10) as response:
                data = response.read()
            return Image.open(io.BytesIO(data)).convert("RGB")
        except (URLError, ValueError, OSError):
            st.sidebar.error("Failed to load image from URL.")

    return None


def draw_detections(
    image: Image.Image,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    categories: Sequence[str],
    alpha: float,
    masks: Optional[np.ndarray] = None,
    mask_threshold: float = 0.5,
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)

    if masks is not None and len(masks) > 0:
        for i in range(len(masks)):
            mask_data = masks[i, 0]
            mask_binary = mask_data >= mask_threshold
            if not np.any(mask_binary):
                continue
            color = np.array(color_for_index(int(labels[i])), dtype=np.float32)
            base[mask_binary] = (1.0 - alpha) * base[mask_binary] + alpha * color

    output = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(output)

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = color_for_index(int(label))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        title = f"{safe_category_name(int(label), categories)} {float(score):.2f}"
        draw.text((x1 + 3, max(0, y1 - 14)), title, fill=color)

    return output


def filter_predictions(
    prediction: Dict[str, torch.Tensor],
    score_threshold: float,
    max_detections: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    scores = prediction["scores"].detach().cpu().numpy()
    boxes = prediction["boxes"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()

    order = np.argsort(scores)[::-1]
    keep = [idx for idx in order if scores[idx] >= score_threshold]
    keep = keep[:max_detections]

    if not keep:
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        empty_labels = np.empty((0,), dtype=np.int64)
        empty_scores = np.empty((0,), dtype=np.float32)
        return empty_boxes, empty_labels, empty_scores, None

    filtered_masks = None
    if "masks" in prediction:
        masks = prediction["masks"].detach().cpu().numpy()
        filtered_masks = masks[keep]

    return boxes[keep], labels[keep], scores[keep], filtered_masks


def run_fcn(
    image: Image.Image,
    alpha: float,
    chosen_classes: Sequence[str],
) -> Tuple[Image.Image, Dict[str, float], pd.DataFrame]:
    model, weights = load_fcn_model()
    categories = list(weights.meta.get("categories", []))
    transform = weights.transforms()

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    start = time.perf_counter()
    with torch.inference_mode():
        prediction = model(image_tensor)["out"][0]
    latency_ms = (time.perf_counter() - start) * 1000.0

    seg_map = prediction.argmax(0).detach().cpu().numpy()
    base = np.asarray(image.convert("RGB"), dtype=np.float32)

    selected_ids = set()
    if chosen_classes:
        index_lookup = {name: idx for idx, name in enumerate(categories)}
        selected_ids = {index_lookup[c] for c in chosen_classes if c in index_lookup}

    unique_ids = [idx for idx in np.unique(seg_map) if idx != 0]
    if selected_ids:
        unique_ids = [idx for idx in unique_ids if idx in selected_ids]

    class_rows = []
    total_pixels = seg_map.shape[0] * seg_map.shape[1]

    for class_idx in unique_ids:
        mask = seg_map == class_idx
        ratio = float(np.sum(mask)) / float(total_pixels)
        class_rows.append(
            {
                "Class": safe_category_name(int(class_idx), categories),
                "Pixel ratio": round(ratio * 100.0, 2),
            }
        )

        color = np.array(color_for_index(int(class_idx)), dtype=np.float32)
        base[mask] = (1.0 - alpha) * base[mask] + alpha * color

    out_image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    summary = pd.DataFrame(class_rows).sort_values("Pixel ratio", ascending=False)

    stats = {
        "latency_ms": latency_ms,
        "detected_classes": float(len(unique_ids)),
    }
    return out_image, stats, summary


def run_faster_rcnn(
    image: Image.Image,
    score_threshold: float,
    max_detections: int,
    alpha: float,
) -> Tuple[Image.Image, Dict[str, float]]:
    model, weights, categories = load_faster_rcnn_model()
    transform = weights.transforms()
    image_tensor = transform(image).to(DEVICE)

    start = time.perf_counter()
    with torch.inference_mode():
        prediction = model([image_tensor])[0]
    latency_ms = (time.perf_counter() - start) * 1000.0

    boxes, labels, scores, _ = filter_predictions(
        prediction,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )

    result = draw_detections(
        image,
        boxes,
        labels,
        scores,
        categories=categories,
        alpha=alpha,
    )

    stats = {
        "latency_ms": latency_ms,
        "detections": float(len(scores)),
    }
    return result, stats


def run_fast_rcnn_proxy(
    image: Image.Image,
    score_threshold: float,
    max_detections: int,
    alpha: float,
) -> Tuple[Image.Image, Dict[str, float]]:
    model, weights, categories = load_fast_proxy_model()
    transform = weights.transforms()
    image_tensor = transform(image).to(DEVICE)

    start = time.perf_counter()
    with torch.inference_mode():
        prediction = model([image_tensor])[0]
    latency_ms = (time.perf_counter() - start) * 1000.0

    boxes, labels, scores, _ = filter_predictions(
        prediction,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )

    result = draw_detections(
        image,
        boxes,
        labels,
        scores,
        categories=categories,
        alpha=alpha,
    )

    stats = {
        "latency_ms": latency_ms,
        "detections": float(len(scores)),
    }
    return result, stats


def run_rcnn_demo(
    image: Image.Image,
    score_threshold: float,
    max_detections: int,
    alpha: float,
    proposal_threshold: float,
    proposal_budget: int,
) -> Tuple[Image.Image, Dict[str, float]]:
    det_model, det_weights, categories = load_faster_rcnn_model()
    cls_model, cls_weights = load_region_classifier()

    det_transform = det_weights.transforms()
    cls_transform = cls_weights.transforms()

    image_tensor = det_transform(image).to(DEVICE)

    start_total = time.perf_counter()
    with torch.inference_mode():
        prediction = det_model([image_tensor])[0]

    det_scores = prediction["scores"].detach().cpu().numpy()
    det_boxes = prediction["boxes"].detach().cpu().numpy()
    det_labels = prediction["labels"].detach().cpu().numpy()

    order = np.argsort(det_scores)[::-1]
    candidates = [idx for idx in order if det_scores[idx] >= proposal_threshold]
    candidates = candidates[:proposal_budget]

    if not candidates:
        return image, {"latency_ms": (time.perf_counter() - start_total) * 1000.0, "detections": 0.0}

    proposal_boxes = det_boxes[candidates]
    proposal_labels = det_labels[candidates]
    proposal_scores = det_scores[candidates]

    crops = []
    valid_positions = []
    width, height = image.size

    for i, box in enumerate(proposal_boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            continue

        crop = image.crop((x1, y1, x2, y2)).convert("RGB")
        crops.append(cls_transform(crop))
        valid_positions.append(i)

    cls_conf = np.zeros(len(proposal_boxes), dtype=np.float32)
    if crops:
        batch = torch.stack(crops).to(DEVICE)
        with torch.inference_mode():
            logits = cls_model(batch)
            probs = torch.softmax(logits, dim=1)
            max_probs = probs.amax(dim=1).detach().cpu().numpy()
        cls_conf[valid_positions] = max_probs

    merged_scores = 0.7 * proposal_scores + 0.3 * cls_conf
    keep = np.where(merged_scores >= score_threshold)[0]

    if len(keep) > 0:
        keep_sorted = keep[np.argsort(merged_scores[keep])[::-1]]
        keep_sorted = keep_sorted[:max_detections]
        final_boxes = proposal_boxes[keep_sorted]
        final_labels = proposal_labels[keep_sorted]
        final_scores = merged_scores[keep_sorted]
    else:
        final_boxes = np.empty((0, 4), dtype=np.float32)
        final_labels = np.empty((0,), dtype=np.int64)
        final_scores = np.empty((0,), dtype=np.float32)

    result = draw_detections(
        image,
        final_boxes,
        final_labels,
        final_scores,
        categories=categories,
        alpha=alpha,
    )

    latency_ms = (time.perf_counter() - start_total) * 1000.0
    stats = {
        "latency_ms": latency_ms,
        "detections": float(len(final_scores)),
        "proposals": float(len(candidates)),
    }
    return result, stats


def run_mask_rcnn(
    image: Image.Image,
    score_threshold: float,
    max_detections: int,
    alpha: float,
    mask_threshold: float,
) -> Tuple[Image.Image, Dict[str, float]]:
    model, weights, categories = load_mask_rcnn_model()
    transform = weights.transforms()
    image_tensor = transform(image).to(DEVICE)

    start = time.perf_counter()
    with torch.inference_mode():
        prediction = model([image_tensor])[0]
    latency_ms = (time.perf_counter() - start) * 1000.0

    boxes, labels, scores, masks = filter_predictions(
        prediction,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )

    result = draw_detections(
        image,
        boxes,
        labels,
        scores,
        categories=categories,
        alpha=alpha,
        masks=masks,
        mask_threshold=mask_threshold,
    )

    stats = {
        "latency_ms": latency_ms,
        "detections": float(len(scores)),
    }
    return result, stats


def benchmark_selected_methods(
    image: Image.Image,
    methods: Sequence[str],
    runs: int,
    score_threshold: float,
    max_detections: int,
    alpha: float,
    mask_threshold: float,
    rcnn_proposal_threshold: float,
    rcnn_proposal_budget: int,
) -> pd.DataFrame:
    results = []

    for method in methods:
        latencies = []
        counts = []

        for _ in range(runs):
            if method == "FCN":
                _, stats, _ = run_fcn(image, alpha=alpha, chosen_classes=[])
            elif method == "R-CNN Demo":
                _, stats = run_rcnn_demo(
                    image,
                    score_threshold=score_threshold,
                    max_detections=max_detections,
                    alpha=alpha,
                    proposal_threshold=rcnn_proposal_threshold,
                    proposal_budget=rcnn_proposal_budget,
                )
            elif method == "Fast R-CNN Proxy":
                _, stats = run_fast_rcnn_proxy(
                    image,
                    score_threshold=score_threshold,
                    max_detections=max_detections,
                    alpha=alpha,
                )
            elif method == "Faster R-CNN":
                _, stats = run_faster_rcnn(
                    image,
                    score_threshold=score_threshold,
                    max_detections=max_detections,
                    alpha=alpha,
                )
            elif method == "Mask R-CNN":
                _, stats = run_mask_rcnn(
                    image,
                    score_threshold=score_threshold,
                    max_detections=max_detections,
                    alpha=alpha,
                    mask_threshold=mask_threshold,
                )
            else:
                continue

            latencies.append(float(stats.get("latency_ms", 0.0)))
            counts.append(float(stats.get("detections", stats.get("detected_classes", 0.0))))

        if not latencies:
            continue

        avg_latency = float(np.mean(latencies))
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        results.append(
            {
                "Method": method,
                "Avg latency (ms)": round(avg_latency, 2),
                "FPS": round(fps, 2),
                "Avg objects/classes": round(float(np.mean(counts)), 2),
            }
        )

    return pd.DataFrame(results)


def render_image_pair(before: Image.Image, after: Image.Image):
    c1, c2 = st.columns(2)
    with c1:
        st.image(before, caption="Before", use_container_width=True)
    with c2:
        st.image(after, caption="After", use_container_width=True)


def render_stats(stats: Dict[str, float], include_proposals: bool = False):
    cols = st.columns(3)
    cols[0].metric("Latency (ms)", f"{stats.get('latency_ms', 0.0):.2f}")
    cols[1].metric("FPS", f"{(1000.0 / stats.get('latency_ms', 1.0)):.2f}")
    cols[2].metric("Objects / classes", f"{stats.get('detections', stats.get('detected_classes', 0.0)):.0f}")

    if include_proposals:
        st.caption(f"R-CNN proposal budget used: {int(stats.get('proposals', 0.0))}")


st.title("Interactive Image Processor with FCN / R-CNN Family / Mask R-CNN")
st.caption(
    "Real-time parameter tuning with before/after comparison. "
    "R-CNN and Fast R-CNN are educational demos; Faster R-CNN and Mask R-CNN use official pretrained models from torchvision."
)

st.sidebar.header("Global Controls")
mode = st.sidebar.selectbox(
    "Select module",
    options=[
        "1) FCN Semantic Segmentation",
        "2) R-CNN / Fast R-CNN / Faster R-CNN",
        "3) Mask R-CNN Instance Segmentation",
        "4) Performance Comparison",
    ],
)

score_threshold = st.sidebar.slider("Score threshold", 0.05, 0.95, 0.50, 0.05)
mask_threshold = st.sidebar.slider("Mask threshold", 0.10, 0.95, 0.50, 0.05)
alpha = st.sidebar.slider("Overlay alpha", 0.10, 0.90, 0.45, 0.05)
max_detections = st.sidebar.slider("Max detections", 1, 100, 25, 1)

image = load_uploaded_or_url_image()

st.sidebar.markdown("---")
st.sidebar.write(f"Device: **{DEVICE.type.upper()}**")
st.sidebar.write("Tip: on CPU, first inference can be slow because models are downloaded and warmed up.")

if image is None:
    st.info("Upload an image or provide a URL to start the demo.")
    st.stop()

if mode == "1) FCN Semantic Segmentation":
    st.subheader("FCN Semantic Segmentation")
    categories = FCN_ResNet50_Weights.DEFAULT.meta.get("categories", [])
    selectable_classes = [c for i, c in enumerate(categories) if i != 0]

    selected_classes = st.multiselect(
        "Optional class focus (leave empty to show all detected classes)",
        options=selectable_classes,
        default=[],
    )

    with st.spinner("Running FCN segmentation..."):
        result_image, stats, class_summary = run_fcn(
            image,
            alpha=alpha,
            chosen_classes=selected_classes,
        )

    render_image_pair(image, result_image)
    render_stats(stats)

    if not class_summary.empty:
        st.dataframe(class_summary, use_container_width=True)
    else:
        st.warning("No foreground classes found for the selected filter.")

elif mode == "2) R-CNN / Fast R-CNN / Faster R-CNN":
    st.subheader("R-CNN Family Object Detection")
    variant = st.radio(
        "Detector variant",
        options=["R-CNN Demo", "Fast R-CNN Proxy", "Faster R-CNN"],
        horizontal=True,
    )

    rcnn_proposal_threshold = st.slider("R-CNN proposal threshold", 0.01, 0.60, 0.10, 0.01)
    rcnn_proposal_budget = st.slider("R-CNN proposal budget", 4, 96, 20, 2)

    with st.spinner(f"Running {variant}..."):
        if variant == "R-CNN Demo":
            result_image, stats = run_rcnn_demo(
                image,
                score_threshold=score_threshold,
                max_detections=max_detections,
                alpha=alpha,
                proposal_threshold=rcnn_proposal_threshold,
                proposal_budget=rcnn_proposal_budget,
            )
        elif variant == "Fast R-CNN Proxy":
            result_image, stats = run_fast_rcnn_proxy(
                image,
                score_threshold=score_threshold,
                max_detections=max_detections,
                alpha=alpha,
            )
        else:
            result_image, stats = run_faster_rcnn(
                image,
                score_threshold=score_threshold,
                max_detections=max_detections,
                alpha=alpha,
            )

    render_image_pair(image, result_image)
    render_stats(stats, include_proposals=(variant == "R-CNN Demo"))

    st.markdown(
        "**Note**: R-CNN and Fast R-CNN are simplified educational implementations in this app. "
        "Faster R-CNN uses the official torchvision pretrained detector."
    )

elif mode == "3) Mask R-CNN Instance Segmentation":
    st.subheader("Mask R-CNN Instance Segmentation")

    with st.spinner("Running Mask R-CNN..."):
        result_image, stats = run_mask_rcnn(
            image,
            score_threshold=score_threshold,
            max_detections=max_detections,
            alpha=alpha,
            mask_threshold=mask_threshold,
        )

    render_image_pair(image, result_image)
    render_stats(stats)

else:
    st.subheader("Performance Comparison")
    st.write(
        "Compare local runtime under current settings and reference paper-level numbers. "
        "For CPU environments, keep runs low to reduce waiting time."
    )

    rcnn_proposal_threshold = st.slider("Benchmark R-CNN proposal threshold", 0.01, 0.60, 0.10, 0.01)
    rcnn_proposal_budget = st.slider("Benchmark R-CNN proposal budget", 4, 96, 20, 2)

    methods = st.multiselect(
        "Methods to benchmark",
        options=["FCN", "R-CNN Demo", "Fast R-CNN Proxy", "Faster R-CNN", "Mask R-CNN"],
        default=["FCN", "R-CNN Demo", "Fast R-CNN Proxy", "Faster R-CNN", "Mask R-CNN"],
    )
    runs = st.slider("Benchmark runs per method", 1, 3, 1, 1)

    if st.button("Run benchmark", type="primary"):
        with st.spinner("Benchmarking selected methods..."):
            table = benchmark_selected_methods(
                image=image,
                methods=methods,
                runs=runs,
                score_threshold=score_threshold,
                max_detections=max_detections,
                alpha=alpha,
                mask_threshold=mask_threshold,
                rcnn_proposal_threshold=rcnn_proposal_threshold,
                rcnn_proposal_budget=rcnn_proposal_budget,
            )

        if table.empty:
            st.warning("No benchmark result available. Please select at least one method.")
        else:
            st.dataframe(table, use_container_width=True)

            chart = (
                alt.Chart(table)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Method:N", sort="-y"),
                    y=alt.Y("Avg latency (ms):Q"),
                    color=alt.Color("Method:N", legend=None),
                    tooltip=list(table.columns),
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)

    st.markdown("### Reference numbers from literature (non-local runtime)")
    st.table(pd.DataFrame(REFERENCE_METRICS))

st.markdown("---")
st.caption(
    "Deployment target: Streamlit Community Cloud (https://share.streamlit.io). "
    "Push this folder to GitHub and deploy app.py as the entrypoint."
)
