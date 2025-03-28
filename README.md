<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🚗 Introduction

Welcome to the [Ultralytics Velocity](https://github.com/ultralytics/velocity) repository! Here, we delve into the intersection of [Machine Learning (ML)](https://www.ultralytics.com/glossary/machine-learning-ml) and [Structure From Motion (SFM)](https://en.wikipedia.org/wiki/Structure_from_motion) to estimate the speed of vehicles using image analysis. Our objective is to enhance vehicle speed estimation methodologies and provide a foundation for future research and practical applications in fields like [traffic management](https://www.ultralytics.com/blog/ai-in-traffic-management-from-congestion-to-coordination) and [autonomous systems](https://www.ultralytics.com/glossary/autonomous-vehicles).

[![Ultralytics Actions](https://github.com/ultralytics/velocity/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/velocity/actions/workflows/format.yml) <a href="https://discord.com/invite/ultralytics"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a> <a href="https://community.ultralytics.com/"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a> <a href="https://reddit.com/r/ultralytics"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>

## 🎯 Project Objectives

This project aims to leverage advanced [ML](https://www.ultralytics.com/glossary/machine-learning-ml) and SFM techniques to accurately estimate vehicle speeds from various forms of imagery. By developing these methods, we hope to contribute valuable tools applicable to traffic monitoring, [autonomous driving systems](https://www.ultralytics.com/solutions/ai-in-automotive), and road safety analysis.

## 📸 Dataset

Currently, a public [dataset](https://www.ultralytics.com/glossary/benchmark-dataset) is not provided with this repository. The methods are designed for integration with custom datasets. If you possess relevant imagery or wish to collaborate on applying these techniques, please contact us. For general dataset needs, explore resources like [Roboflow](https://roboflow.com/?ref=ultralytics) or public datasets like [COCO](https://docs.ultralytics.com/datasets/detect/coco/).

# 📋 Requirements

To execute the code within this repository, ensure you meet the following prerequisites:

- **Python 3.7+**: Install [Python](https://www.python.org/) and use [pip](https://pip.pypa.io/en/stable/) to set up the necessary libraries:

  ```bash
  pip3 install -U -r requirements.txt
  ```

  The `requirements.txt` file includes essential Python packages such as:

  - [`numpy`](https://numpy.org/): Fundamental package for numerical computation.
  - [`scipy`](https://scipy.org/): Library for scientific and technical computing.
  - [`torch`](https://pytorch.org/): An open-source [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) framework ([PyTorch](https://www.ultralytics.com/glossary/pytorch)).
  - [`opencv-python`](https://pypi.org/project/opencv-python/): [OpenCV](https://opencv.org/) library for [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks.
  - [`exifread`](https://github.com/ianare/exif-py): Library to read Exif metadata from image files.
  - [`bokeh`](https://bokeh.org/) (optional): For interactive data visualization.

- **MATLAB 2018a or newer**: Some scripts require [MATLAB](https://www.mathworks.com/products/matlab.html). Clone the common functions repository and add it to your MATLAB path:

  ```bash
  git clone https://github.com/ultralytics/functions-matlab
  ```

  Then, within MATLAB:

  ```matlab
  >> addpath(genpath('/path/to/functions-matlab')) % Replace /path/to/ with the actual path
  ```

  Ensure the following MATLAB toolboxes are installed:

  - `Statistics and Machine Learning Toolbox`
  - `Signal Processing Toolbox`

# 🏃 Run

This repository offers various methods for vehicle speed estimation using SFM and ML. While detailed run instructions are context-dependent, the core scripts leverage the libraries listed in the requirements. If you're interested in applying these techniques or need specific guidance on execution, please don't hesitate to reach out or raise an [Issue](https://github.com/ultralytics/velocity/issues).

<img src="https://github.com/ultralytics/velocity/blob/main/results.jpg" alt="Sample speed estimation results visualization">

# 📚 Citation

If this repository contributes to your research or project, please cite it using the following DOI:

[![DOI](https://zenodo.org/badge/126519968.svg)](https://zenodo.org/badge/latestdoi/126519968)

# 🤝 Contribute

We actively welcome contributions from the community! Whether it's fixing bugs, proposing new features, or enhancing documentation, your input is highly valued. Please see our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for more details on how to get started. We also encourage you to share your experiences with Ultralytics projects by completing our brief [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). A huge 🙏 thank you to all our contributors!

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

# ©️ License

Ultralytics provides two licensing options to accommodate different use cases:

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/license/agpl-v3) open-source license is ideal for students, researchers, and enthusiasts keen on open collaboration and knowledge sharing. It promotes transparency and community involvement. See the [LICENSE](https://github.com/ultralytics/velocity/blob/main/LICENSE) file for full details.
- **Enterprise License**: Designed for commercial applications, this license permits the seamless integration of Ultralytics software and AI models into commercial products and services, bypassing the open-source requirements of AGPL-3.0. If your project requires commercial licensing, please contact us through [Ultralytics Licensing](https://www.ultralytics.com/license).

# 📬 Contact Us

For bug reports, feature suggestions, and contributions, please visit [GitHub Issues](https://github.com/ultralytics/velocity/issues). For broader questions and discussions about this project or other Ultralytics initiatives, join our vibrant community on [Discord](https://discord.com/invite/ultralytics)!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
