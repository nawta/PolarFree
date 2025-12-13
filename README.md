



# 🚀 **PolarFree: Polarization-based Reflection-Free Imaging** – [CVPR 2025]  



<p align="center">
    <a href="https://arxiv.org/abs/2503.18055"><img src="https://img.shields.io/badge/arXiv-2503.18055-b31b1b.svg" alt="arXiv Paper"></a>
    <a href="https://mdyao.github.io/PolarFree/"><img src="https://img.shields.io/badge/Project-Page-brightgreen.svg" alt="Project Page"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg" alt="PyTorch Version"></a>
    <a href="https://github.com/mdyao/PolarFree/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
    <a href="https://huggingface.co/datasets/Mingde/PolaRGB"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue" alt="Hugging Face Dataset"></a>
    <a href="https://youtu.be/g-aNjbygiJM"><img src="https://img.shields.io/badge/YouTube-Demo-red.svg" alt="YouTube Demo"></a>
</p>

🌟 *A Solution and Dataset for Polarization-based Reflection-Free Imaging*  

<img src="https://raw.githubusercontent.com/mdyao/PolarFree/doc/docs/static/images/reflection-refraction-polarization.gif" alt="Polarization-based Reflection and Refraction" width="50%">


*Image source: [ThinkLucid](https://thinklucid.com/tech-briefs/polarization-explained-sony-polarized-sensor/)*  


---


## 🎥 **Demo Video**
Watch the demonstration of PolarFree in action:  
[![PolarFree Demo](https://img.youtube.com/vi/g-aNjbygiJM/0.jpg)](https://youtu.be/g-aNjbygiJM)

---

## 📌 **Highlights**  
✅ **Large-Scale Dataset**: PolaRGB includes 6,500 well-aligned RGB-polarization image pairs, 8× larger than existing datasets.  
✅ **Innovative Method**: PolarFree leverages diffusion models to generate reflection-free priors for accurate reflection removal.  
✅ **State-of-the-Art Performance**: Outperforms existing methods by ~2dB in PSNR on challenging real-world scenarios.  
✅ **Open Source**: Code and dataset are freely available for research and development.  


## ⏳ **Timeline**  

- ✅ **2025-03-23** - 🛠️ Repository initialized with documentation.  
- ✅ **2025-03-23** - 🔗 Project Page officially launched.  
- ✅ **2025-03-23** - 📄 Paper available on arXiv.  
- ✅ **2025-04-21** - 🚀 Provide core codebase, testing subset, and pre-trained models for evaluation.  
- ✅ **2025-06-28** - 📦 Release the full PolaRGB dataset with download links.  
- ✅ **2025-06-28** - 📝 Publish training code.  
- TODO: Provide detailed training and testing instructions.

## 📖 **Overview**  
PolarFree addresses the challenging task of reflection removal using polarization cues and a novel diffusion-based approach. Key contributions include:  
- **PolaRGB Dataset**: A large-scale dataset with diverse indoor and outdoor scenes, providing RGB and polarization images.  

![Dataset Overview](https://raw.githubusercontent.com/mdyao/PolarFree/doc/docs/static/images/dataset_overview.png)

- **Diffusion Model**: Utilizes diffusion processes to generate reflection-free priors, enabling precise reflection removal and improved image clarity.  
![Model Design](https://raw.githubusercontent.com/mdyao/PolarFree/doc/docs/static/images/model_design.png)

- **Superior Results**: Extensive experiments on the PolaRGB dataset show that PolarFree outperforms existing methods by ~2dB in PSNR, achieving cleaner reflection removal and sharper image details.  

- **Real-World Effectiveness**: PolarFree demonstrates robust performance in real-world scenarios, such as museums and galleries, effectively reducing reflections while preserving fine details.  



---

## 🚀 **Installation & Usage**

### 1. Clone the Repository

```bash
git clone https://github.com/mdyao/PolarFree.git
cd PolarFree
```

## Requirements

Due to the complexity of my development environment, I do not provide a complete `requirements.txt` file.  
However, the following key dependencies and their versions are provided for reference:

- basicsr == 1.4.2  
- mmcv == 2.1.0  
- torch == 2.4.1  
- torchvision == 0.19.1


### 2. Download the Dataset

You can access the dataset from Hugging Face:  
👉 [https://huggingface.co/datasets/Mingde/PolaRGB](https://huggingface.co/datasets/Mingde/PolaRGB)

If you want raw images, you can access the raw image on 👉 [https://huggingface.co/datasets/Mingde/PolaRGB_raw](https://huggingface.co/datasets/Mingde/PolaRGB_raw)

Download and organize the dataset according to the structure required by the codebase.



### 3. Run the Demo

Once everything is set up, run the demo script:

```bash
python simple_test.py -opt options/test/test.yml -gpu_id 0
```
--- 

## 📊 **Results**
PolarFree achieves superior performance compared to existing methods:

![Results](https://raw.githubusercontent.com/mdyao/PolarFree/doc/docs/static/images/results.png)

## 📜 **Citation**
If you find this work useful, please cite:

    @inproceedings{polarfree2025,
      title   = {PolarFree: Polarization-based Reflection-Free Imaging},
      author  = {Yao, Mingde and Wang, Menglu and Tam, King-Man and Li, Lingen and Xue, Tianfan and Gu, Jinwei},
      booktitle = {CVPR},
      year    = {2025},
    }



<!--    
<p align="center">  
  <img src="docs/banner.png" alt="Project Banner" width="80%">  
</p>  

Project Page]()🔗 [**Paper**](https://arxiv.org/abs/xxxxx) | 📦 [**Dataset**](Coming soon) | | 🎥 [**Video**](https://xxxx)  

---

## 📌 **Highlights**  
✅ **State-of-the-art**: Outperforms existing methods on [benchmark].  
✅ **Fast & Efficient**: Achieves [metric] improvement with [speedup] performance.  
✅ **Easy to Use**: Plug & play implementation with PyTorch.  
✅ **Open-Source & Reproducible**: Code, dataset, and pre-trained models are freely available.  

---

## ⏳ **Timeline**  
📅 *Key Milestones in Our Research Journey*  

- **YYYY-MM-DD** - 📝 Paper submitted to CVPR 202X.  
- **YYYY-MM-DD** - ✅ Paper accepted at CVPR 202X.  
- **YYYY-MM-DD** - 📢 Preprint available on [arXiv].  
- **YYYY-MM-DD** - 📦 Code and dataset released on GitHub.  
- **YYYY-MM-DD** - 🚀 Added new features & improvements.  

---

## 📖 **Overview**  
🔍 *A brief introduction to your project.*  

- **Goal**: Solve [problem] using [method].  
- **Method**: Uses [techniques] with [model/architecture].  
- **Results**: Achieves [SOTA results] on [benchmark].  

---

## 🚀 **Installation**  
```bash
git clone https://github.com/your-repo/project-name.git
cd project-name
pip install -r requirements.txt
```

---

## 🏁 **Quick Start**  
```bash
python demo.py --input example.jpg --output result.jpg
```

---

## 📊 **Results & Comparisons**  
📌 *Showcase performance metrics, comparisons with SOTA, and visual results.*  

| Method | Dataset | Accuracy | Speed |
|--------|--------|---------|-------|
| **Ours** | [Dataset] | **XX%** | **XX ms** |
| Baseline | [Dataset] | XX% | XX ms |

---

## 📜 **Citation**  
If you find this work useful, please cite:  
```bibtex
@inproceedings{your_paper,
  title={Your Paper Title},
  author={Your Name and Co-authors},
  booktitle={CVPR},
  year={202X}
}
```

---

## 📝 **License**  
This project is released under the [MIT License](LICENSE).  

🙌 **Star** ⭐ and **Fork** 🍴 this repo if you find it useful! 🚀  

---
 -->
