<br>
<a href="https://ultralytics.com" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🚗 Introduction

Welcome to the [Ultralytics Velocity](https://github.com/ultralytics/velocity) repository! Here, we're exploring the frontiers of Machine Learning (ML) and Structure From Motion (SFM) to estimate the speed of vehicles using imagery analysis. Our goal is to advance vehicle speed estimation techniques and provide a springboard for further research and application in this area.

[![Ultralytics Actions](https://github.com/ultralytics/velocity/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/velocity/actions/workflows/format.yml) <a href="https://ultralytics.com/discord"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a> <a href="https://community.ultralytics.com"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a>

## 🎯 Project Objectives

This project aims to utilize advanced ML and SFM approaches to accurately estimate vehicle speeds from various forms of imagery. By doing so, we hope to contribute valuable tools and methods that can be applied in domains such as traffic monitoring, autonomous driving systems, and road safety enhancements.

## 📸 Dataset

Currently, we do not provide a public dataset alongside this repository; the methods are tailored for use with custom datasets. If you have specific imagery or are interested in applying these techniques, please reach out for collaboration.

# 📋 Requirements

To run the code in this repository, you will need the following:

- **Python 3.7+**: Install Python and use pip to set up the necessary libraries:

  ```
  pip3 install -U -r requirements.txt
  ```

  The `requirements.txt` file includes key Python packages such as:

  - `numpy`
  - `scipy`
  - `torch` (PyTorch)
  - `opencv-python` (OpenCV)
  - `exifread`
  - `bokeh` (optional, for visualization)

- **MATLAB 2018a or newer**: Besides Python, some scripts require MATLAB. Clone the common functions repository and add it to your MATLAB path with the following commands:

  ```
  git clone https://github.com/ultralytics/functions-matlab
  ```

  Then in MATLAB:

  ```
  >> addpath(genpath('/functions-matlab'))
  ```

  Ensure you have the following MATLAB toolboxes installed:

  - `Statistics and Machine Learning Toolbox`
  - `Signal Processing Toolbox`

# 🏃 Run

The repository contains various methods for vehicle speed estimation. If you're interested in leveraging these techniques or require more information on running the code, please feel free to reach out.

<img src="https://github.com/ultralytics/velocity/blob/main/results.jpg">

# 📚 Citation

If our work assists you in your research or project, please consider citing it using the following DOI:

[![DOI](https://zenodo.org/badge/126519968.svg)](https://zenodo.org/badge/latestdoi/126519968)

# 🤝 Contribute

We welcome contributions from the community! Whether you're fixing bugs, adding new features, or improving documentation, your input is invaluable. Take a look at our [Contributing Guide](https://docs.ultralytics.com/help/contributing) to get started. Also, we'd love to hear about your experience with Ultralytics products. Please consider filling out our [Survey](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). A huge 🙏 and thank you to all of our contributors!

<a href="https://github.com/ultralytics/yolov5/graphs/contributors">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/image-contributors.png" alt="Ultralytics open-source contributors"></a>

# ©️ License

Ultralytics is excited to offer two different licensing options to meet your needs:

- **AGPL-3.0 License**: Perfect for students and hobbyists, this [OSI-approved](https://opensource.org/licenses/) open-source license encourages collaborative learning and knowledge sharing. Please refer to the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for detailed terms.
- **Enterprise License**: Ideal for commercial use, this license allows for the integration of Ultralytics software and AI models into commercial products without the open-source requirements of AGPL-3.0. For use cases that involve commercial applications, please contact us via [Ultralytics Licensing](https://ultralytics.com/license).

# 📬 Contact Us

For bug reports, feature requests, and contributions, head to [GitHub Issues](https://github.com/ultralytics/velocity/issues). For questions and discussions about this project and other Ultralytics endeavors, join us on [Discord](https://ultralytics.com/discord)!

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
  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
