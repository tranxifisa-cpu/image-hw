# Streamlit Image Processor (FCN / R-CNN Family / Mask R-CNN)

This project is a Streamlit web app with real-time parameter tuning, before/after visual comparison, and method benchmarking for:

1. FCN semantic segmentation demo
2. R-CNN / Fast R-CNN / Faster R-CNN object detection demos
3. Mask R-CNN instance segmentation demo
4. Runtime and reference-metric comparison

## Features

- Real-time sliders for score threshold, mask threshold, overlay alpha, and max detections
- Side-by-side image comparison (before vs after)
- FCN class-level segmentation summary
- R-CNN family demo panel (including educational R-CNN and Fast R-CNN approximations)
- Benchmark panel with latency/FPS table and chart
- Reference paper metrics table for quick method comparison

## Run Locally

```bash
cd image5
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit (usually `http://localhost:8501`).

## Deploy to Streamlit Community Cloud

Target platform: https://share.streamlit.io/

1. Push this `image5` folder into a GitHub repository.
2. Go to Streamlit Community Cloud and sign in with GitHub.
3. Create a new app and set:
   - Repository: `<your-github-user>/<your-repo>`
   - Branch: usually `main`
   - Main file path: `image5/app.py`
4. Click Deploy.

Your public URL will be:

```text
https://<your-app-name>.streamlit.app
```

## Notes

- The first run may take longer because model weights are downloaded.
- CPU mode is supported, but inference can be slower than GPU mode.
- R-CNN and Fast R-CNN are educational approximations in this app.
- Faster R-CNN and Mask R-CNN use official torchvision pretrained models.
