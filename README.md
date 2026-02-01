# Project aim
This project compares and benchmarks custom data preprocessing techniques for Computer Vision using the Jester dataset. The goal was to measure how different data strategies impact model accuracy and to test the feasibility of using 2D CNNs with optimized preprocessing as a resource-efficient alternative to complex 3D models.

We did not aim for state-of-the-art accuracy, but rather focused on the delta in performance caused by preprocessing. Specifically, we investigated if a lightweight 2D model (using frame averaging strategies) could compete with computationally expensive 3D models (using stacked frames) in scenarios with limited data and compute resources.

# Methods
We trained 6 models in total: 3 baseline 2D models and 3 3D models.

## Preprocessing Strategies
For the 2D baseline, we collapsed the temporal dimension into a single image. For the 3D approach, we stacked the frames to preserve temporal depth.
- Simple Mean: averaging raw frame information. This resulted in a static image showing the "shadow" of the movement.
- Absolute Difference: calculating the absolute difference between the first frame and every subsequent frame, then averaging. This effectively removed the background to highlight horizontal and vertical motion, but lacked depth.
- Relative Difference: subtracting the previous frame from the current one before averaging. This captured depth movement too (e.g. distinguishing between pulling fingers in vs. pushing out).

## Architecture and Training
- 2D Baseline: A TinyVGG architecture adapted for 2D image input.
- 3D Model: A ResNet-20 architecture adapted for 3D spatiotemporal data.
- Optimization: We used Momentum Gradient Descent with a CrossEntropy loss function and a CosineAnnealing scheduler. Best model states were saved based on validation accuracy.
- Efficiency: To reduce training time, we implemented a caching system for every generated image and stack. This reduced training time by approximately *50-66%*.

# Repository Structure 
To navigate the code, please use these key files:
- `full_baseline_pipeline_as_py.py` for generating the 2D data and training the 3 baseline models.
- `3dconv_pipeline` for generating the 3D blocks and training the 3 3D models.

The project was made with a mixed Linux/Windows workspace, so it uses our profiles for determining the proper local file structure to use. Having the raw Jester dataset, or a subset as we used, downloaded is paramount for replication. The rest of the files are simply for exploratory data analysis early in the project, or previous versions of main files with changed names.

# Results
The benchmarking results highlight a significant performance gap depending on the preprocessing strategy used.

| Strategy | 2D Baseline Accuracy | 3D Model Accuracy |
| :--- | :--- | :--- |
| **Simple Mean** | 44% | 80% |
| **Absolute Difference** | 56% | 83% |
| **Relative Difference** | **64%** | **84%** |

While the 2D Simple Mean baseline performed poorly (44%), applying the Relative Difference strategy boosted the 2D model's accuracy to 64%. This approaches the performance territory of the much heavier 3D models (84%).

# Conclusion
This project demonstrates that while 3D models generally outperform 2D baselines, the gap can be significantly narrowed by using optimal data preprocessing. By utilizing the Relative Difference strategy, we achieved a **20%** increase in accuracy for the simple 2D model compared to the naive mean approach. This suggests that for resource-constrained environments, smart data preprocessing is a *viable alternative* to increasing model complexity.
