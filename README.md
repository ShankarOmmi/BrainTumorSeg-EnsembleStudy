# Brain-Tumor-Segmentation-Ensembling

## Overview
This project focuses on brain tumor segmentation using deep learning models. Rather than solely aiming for the highest accuracy, the primary objective is to conduct a comparative study and gain insights into the effectiveness of different models and loss functions. The approach involves training multiple segmentation models and ensembling them to analyze their strengths and weaknesses.

The models used include:

- **Self-defined U-Net**
- **U-Net with ResNet34 encoder (imported)**
- **DeepLabV3+ with ResNet50 encoder (imported)**
- **U-Net++ (used for ensembling final predictions)**

The performance of different loss functions, such as **Binary Cross-Entropy (BCE) Loss**, **Dice Loss**, and a **Hybrid Loss (70% BCE + 30% Dice)**, was evaluated.

## Dataset
- **Task:** Brain tumor segmentation
- **Input:** MRI images
- **Output:** Binary segmentation masks (tumor vs. non-tumor)
- **Preprocessing:** Resizing, normalization

## Methodology
### 1. **Training Individual Models**
#### **Self-defined U-Net:**
- Trained using **BCE Loss** initially.
- Further trained using a **Hybrid Loss (70% BCE + 30% Dice Loss)** to improve segmentation performance.

#### **Imported U-Net with ResNet34 Backbone:**
```python
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model = smp.Unet(
    encoder_name="resnet34",  # ResNet34 as encoder
    encoder_weights="imagenet",  # Pre-trained on ImageNet
    in_channels=3,  # RGB image input
    classes=1  # Binary segmentation
).to(device)
```
- The **encoder** (ResNet34) is pre-trained on ImageNet.
- The **decoder** (segmentation head) is trained from scratch.

#### **Imported DeepLabV3+ with ResNet50 Backbone:**
```python
deeplabv3_model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(device)
```
- The **encoder** (ResNet50) is pre-trained on ImageNet.
- The **decoder** is trained from scratch for segmentation.

### 2. **Ensembling for Improved Segmentation**
- Predictions from **U-Net with ResNet34** and **DeepLabV3+** were used as inputs to a **U-Net++** model.
- The U-Net++ model was trained using both **BCE Loss** and **Hybrid Loss**.
- **Observation:** The U-Net++ model performed **better with normal BCE Loss** compared to Hybrid Loss.

## Results
| Model | Loss Function | Performance |
|--------|--------------|-------------|
| Self-defined U-Net | BCE Loss | Baseline |
| Self-defined U-Net | Hybrid Loss (70% BCE + 30% Dice) | Improved segmentation |
| U-Net (ResNet34) | BCE Loss | Strong feature extraction |
| DeepLabV3+ (ResNet50) | BCE Loss | Best standalone performance |
| U-Net++ (Ensemble) | BCE Loss | Best overall segmentation |
| U-Net++ (Ensemble) | Hybrid Loss | Performed worse than BCE Loss |

Key Takeaway:

- DeepLabV3+ with ResNet50 performed best as a standalone segmentation model.

- Ensembling with U-Net++ further enhanced segmentation performance, demonstrating the benefits of combining multiple models.

- This project is focused on understanding and comparing different models and loss functions rather than just maximizing accuracy. The results provide insights into how ensembling and different loss formulations impact brain tumor segmentation performance.

## Future Work
- Experimenting with other backbone architectures (EfficientNet, DenseNet)
- Applying semi-supervised learning to improve segmentation with limited labels
- Extending the approach to multi-class tumor segmentation

## Contributors
- **[Shankar Ommi](https://github.com/ShankarOmmi)**

## License
This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

