"""
RMBG-2.0 背景移除工具

基于Hugging Face Transformers的RMBG-2.0模型实现的图像背景移除工具。
提供Gradio Web界面，支持图片上传、参数调节和实时处理。

作者: [Your Name]
版本: 1.0.0
依赖: torch, transformers, gradio, PIL, numpy
"""

from typing import Union, Tuple, Optional, Literal
import gradio as gr
from PIL import Image
import torch
from torch import Tensor
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import numpy as np

# 类型别名定义
DeviceType = Literal["cuda", "cpu"]
ImageInputType = Union[Image.Image, np.ndarray, None]
ProcessResultType = Tuple[Optional[Image.Image], str]


def remove_background(
    image: Image.Image, confidence_threshold: float = 0.5, device: DeviceType = "cuda"
) -> Image.Image:
    """
    使用RMBG-2.0模型移除图片背景

    该函数加载预训练的RMBG-2.0模型，对输入图像进行背景分割，
    并返回带有透明背景的PNG图像。

    Args:
        image: 输入的PIL图像对象，支持RGB格式
        confidence_threshold: 置信度阈值，范围[0.1, 0.9]，控制分割的精确度
                            值越高分割越精确，但可能保留更多背景细节
        device: 处理设备，'cuda'使用GPU加速，'cpu'使用CPU处理

    Returns:
        Image.Image: 移除背景后的图像，带有透明通道(RGBA)

    Raises:
        RuntimeError: 当模型加载失败或推理过程中出现错误时
        ValueError: 当输入参数无效时

    Example:
        >>> from PIL import Image
        >>> image = Image.open("input.jpg")
        >>> result = remove_background(image, confidence_threshold=0.6, device="cuda")
        >>> result.save("output.png")

    Note:
        - 首次调用时会下载RMBG-2.0模型文件（约885MB）
        - 建议使用GPU处理以获得更好的性能
        - 输入图像会自动调整到1024x1024进行处理
    """
    # 参数验证
    if not isinstance(image, Image.Image):
        raise ValueError("输入图像必须是PIL.Image.Image对象")

    if not (0.1 <= confidence_threshold <= 0.9):
        raise ValueError("置信度阈值必须在0.1到0.9之间")

    if device not in ["cuda", "cpu"]:
        raise ValueError("设备必须是'cuda'或'cpu'")

    try:
        # 加载RMBG-2.0模型
        model: AutoModelForImageSegmentation = (
            AutoModelForImageSegmentation.from_pretrained(
                "./model/RMBG-2.0", trust_remote_code=True
            )
        )

        # 设置PyTorch精度和模型状态
        torch.set_float32_matmul_precision(["high", "highest"][0])
        model.to(device)
        model.eval()

        # 定义图像预处理流程
        image_size: Tuple[int, int] = (1024, 1024)
        transform_image: transforms.Compose = transforms.Compose(
            [
                transforms.Resize(image_size),  # 调整图像大小
                transforms.ToTensor(),  # 转换为张量 [0,1]
                transforms.Normalize(  # ImageNet标准化
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # 预处理输入图像
        input_tensor: Tensor = transform_image(image).unsqueeze(0).to(device)

        # 模型推理（不计算梯度以节省内存）
        with torch.no_grad():
            # 获取模型输出并应用sigmoid激活函数
            model_output: Tensor = model(input_tensor)[-1]  # 取最后一个输出层
            preds: Tensor = model_output.sigmoid().cpu()  # 转移到CPU进行后处理

        # 提取预测结果
        pred: Tensor = preds[0].squeeze()  # 移除批次维度

        # 应用置信度阈值进行二值化
        pred: Tensor = (pred > confidence_threshold).float()

        # 转换为PIL图像并调整到原始尺寸
        pred_pil: Image.Image = transforms.ToPILImage()(pred)
        mask: Image.Image = pred_pil.resize(image.size)

        # 创建带透明通道的结果图像
        result_image: Image.Image = image.copy()
        result_image.putalpha(mask)

        return result_image

    except Exception as e:
        raise RuntimeError(f"背景移除处理失败: {str(e)}")


def process_image(
    input_image: ImageInputType, confidence_threshold: float, device_choice: DeviceType
) -> ProcessResultType:
    """
    Gradio界面处理函数

    Processing function for Gradio interface

    Args:
        input_image: 用户上传的图像，可能是PIL.Image、numpy数组或None
        confidence_threshold: 置信度阈值，来自Gradio滑块控件
        device_choice: 处理设备选择，来自Gradio单选按钮

    Returns:
        Tuple[Optional[Image.Image], str]:
            - 处理后的图像（失败时为None）
            - 状态信息字符串

    Example:
        >>> result_image, status = process_image(
        ...     pil_image, confidence_threshold=0.5, device_choice="cuda"
        ... )
        >>> print(status)  # "背景移除完成！使用RMBG-2.0模型"
    """
    # 检查输入图像
    if input_image is None:
        return None, "请先上传一张图片"

    try:
        # 统一输入格式：转换为PIL.Image
        if isinstance(input_image, np.ndarray):
            image: Image.Image = Image.fromarray(input_image)
        elif isinstance(input_image, Image.Image):
            image: Image.Image = input_image
        else:
            return None, f"不支持的图像格式: {type(input_image)}"

        # 调用背景移除函数
        result: Image.Image = remove_background(
            image, confidence_threshold, device_choice
        )

        return result, "背景移除完成"

    except Exception as e:
        return None, f"处理过程中出现错误: {str(e)}"


def create_interface() -> gr.Blocks:
    """
    创建Gradio Web界面

    Building a complete Web interface, including image upload, parameter adjustment, processing button, and result display.
    The interface uses a left-right column layout for a better user experience.

    Returns:
        gr.Blocks: Configured Gradio interface object

    Features:
        - Image upload area
        - Parameter adjustment controls (confidence threshold, device selection)
        - Processing button
        - Result display area
        - Usage instructions
        - Example images
    """
    with gr.Blocks(title="RMBG-2.0 背景移除工具", theme=gr.themes.Soft()) as demo:
        # Page title
        gr.Markdown("# 🖼️ RMBG-2.0 背景移除工具")
        gr.Markdown("上传图片，点击开始处理按钮移除背景")

        # Main content area
        with gr.Row():
            # Left input area
            with gr.Column(scale=1):
                # Image upload control
                input_image: gr.Image = gr.Image(label="上传图片", type="pil")

                # Parameter adjustment area
                with gr.Row():
                    confidence_threshold: gr.Slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.05,
                        label="置信度阈值",
                        info="值越高，分割越精确但可能保留更多背景",
                    )

                # Device selection control
                device_choice: gr.Radio = gr.Radio(
                    choices=["cuda", "cpu"],
                    value="cuda",
                    label="处理设备",
                    info="选择GPU或CPU处理",
                )

                # Processing button
                process_btn: gr.Button = gr.Button("🚀 开始处理", variant="primary")

            # Right output area
            with gr.Column(scale=1):
                # Result display
                output_image: gr.Image = gr.Image(label="处理结果", type="pil")
                status_text: gr.Textbox = gr.Textbox(label="状态", interactive=False)

        # Usage instructions
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ## 使用步骤
            
            1. **上传图片**: 点击上传区域选择要处理的图片
            2. **调节参数**: 调整置信度阈值（可选）
            3. **选择设备**: 选择GPU（推荐）或CPU处理
            4. **开始处理**: 点击"开始处理"按钮
            5. **查看结果**: 等待处理完成，查看结果图片
            
            ## 参数说明
            
            - **置信度阈值**: 控制分割的精确度
              - 值越高：分割越精确，但可能保留更多背景细节
              - 值越低：分割更粗糙，但背景移除更彻底
              - 推荐范围：0.4-0.7
            
            - **处理设备**: 
              - GPU (CUDA): 处理速度快，需要NVIDIA显卡
              - CPU: 兼容性好，处理速度较慢
            
            ## 技术说明
            
            - **模型**: 使用RMBG-2.0预训练模型
            - **输入**: 支持JPG、PNG等常见图像格式
            - **输出**: 带透明背景的PNG图像
            - **处理尺寸**: 自动调整到1024x1024进行处理
            
            ## 注意事项
            
            - 首次使用时会下载RMBG-2.0模型文件（约885MB）
            - 建议使用GPU处理以获得更好的性能
            - 处理大图像时可能需要较长时间
            - 结果图像会自动保存为PNG格式
            """)

        # Example images
        gr.Examples(
            examples=[
                [
                    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop"
                ],
                [
                    "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400&h=400&fit=crop"
                ],
            ],
            inputs=input_image,
            label="示例图片",
        )

        # Bind processing function to button click event
        process_btn.click(
            fn=process_image,
            inputs=[input_image, confidence_threshold, device_choice],
            outputs=[output_image, status_text],
        )

    return demo


def main() -> None:
    """
    主函数：启动Gradio Web服务器

    Creating and starting the Gradio interface, configuring server parameters, and providing local and public access addresses.
    """
    # Create interface
    demo: gr.Blocks = create_interface()

    # Start server
    demo.launch(
        server_name="0.0.0.0",  # Listen on all network interfaces
        server_port=7788,  # Server port
        share=False,  # Generate public access link
        show_error=True,  # Show detailed error information
    )


if __name__ == "__main__":
    main()
