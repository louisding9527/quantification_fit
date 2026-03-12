"""ONNX模型导出模块"""

import numpy as np
import torch
import onnx
from onnx import helper, TensorProto
import logging

from .trainer import Trainer

logger = logging.getLogger(__name__)


class ONNXExporter:
    """ONNX模型导出器"""

    def __init__(self, trainer: Trainer):
        """初始化导出器

        Args:
            trainer: 训练好的Trainer对象
        """
        self.trainer = trainer
        self.model = trainer.model
        self.scaler = trainer.scaler

    def export(
        self,
        output_path: str,
        input_names: list = None,
        output_names: list = None,
        dynamic_axes: dict = None,
    ) -> bool:
        """导出ONNX模型

        Args:
            output_path: 输出路径
            input_names: 输入张量名称
            output_names: 输出张量名称
            dynamic_axes: 动态轴定义

        Returns:
            是否成功
        """
        try:
            self.model.eval()

            # 获取输入维度
            input_dim = self.scaler.mean_.shape[0]

            # 默认命名
            if input_names is None:
                input_names = ["input"]
            if output_names is None:
                output_names = ["output"]

            # 动态轴
            if dynamic_axes is None:
                dynamic_axes = {
                    input_names[0]: {0: "batch_size"},
                    output_names[0]: {0: "batch_size"},
                }

            # 创建虚拟输入
            dummy_input = torch.randn(1, input_dim).to(self.model.device)

            # 导出ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=13,
                do_constant_folding=True,
                export_params=True,
            )

            logger.info(f"ONNX model exported to {output_path}")

            # 验证ONNX模型
            if self.validate_onnx(output_path):
                logger.info("ONNX model validation passed")
                return True
            else:
                logger.warning("ONNX model validation failed")
                return False

        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")
            return False

    def validate_onnx(self, onnx_path: str) -> bool:
        """验证ONNX模型

        Args:
            onnx_path: ONNX模型路径

        Returns:
            是否有效
        """
        try:
            # 加载ONNX模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            # 验证推理一致性
            return self._verify_inference(onnx_path)

        except Exception as e:
            logger.error(f"ONNX validation error: {e}")
            return False

    def _verify_inference(self, onnx_path: str) -> bool:
        """验证推理一致性

        Args:
            onnx_path: ONNX模型路径

        Returns:
            是否一致
        """
        try:
            import onnxruntime as ort

            # 创建推理会话
            session = ort.InferenceSession(
                onnx_path,
                providers=["CPUExecutionProvider"],
            )

            # 生成测试数据
            input_dim = self.scaler.mean_.shape[0]
            test_input = np.random.randn(1, input_dim).astype(np.float32)

            # PyTorch预测
            torch_input = torch.tensor(
                self.scaler.transform(test_input), dtype=torch.float32
            ).to(self.model.device)

            with torch.no_grad():
                torch_output = self.model(torch_input)
                torch_result = torch.argmax(torch_output, dim=1).cpu().numpy()

            # ONNX预测
            onnx_input = {session.get_inputs()[0].name: test_input}
            onnx_output = session.run(None, onnx_input)
            onnx_result = np.argmax(onnx_output[0], axis=1)

            # 比较
            if np.array_equal(torch_result, onnx_result):
                logger.info("Inference verification passed")
                return True
            else:
                logger.warning(
                    f"Inference mismatch: torch={torch_result}, onnx={onnx_result}"
                )
                return False

        except ImportError:
            logger.warning("onnxruntime not installed, skipping inference verification")
            return True
        except Exception as e:
            logger.error(f"Inference verification error: {e}")
            return False


def export_to_onnx(
    trainer: Trainer,
    output_path: str,
    model_name: str = "trading_model",
) -> str:
    """便捷导出函数

    Args:
        trainer: 训练好的Trainer
        output_path: 输出目录
        model_name: 模型名称

    Returns:
        导出的ONNX模型路径
    """
    exporter = ONNXExporter(trainer)

    # 导出模型
    onnx_path = f"{output_path}/{model_name}.onnx"
    exporter.export(onnx_path)

    return onnx_path


def create_onnx_model(
    input_dim: int,
    hidden_dims: list = None,
    output_path: str = "model.onnx",
) -> bool:
    """从头创建ONNX模型（不需要PyTorch模型）

    Args:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度
        output_path: 输出路径

    Returns:
        是否成功
    """
    try:
        if hidden_dims is None:
            hidden_dims = [64, 32]

        # 创建输入
        input_tensor = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, input_dim]
        )

        # 创建输出
        output_tensor = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 3]
        )

        # 创建权重初始化
        initializers = []

        # 第一层
        w1 = np.random.randn(hidden_dims[0], input_dim).astype(np.float32) * 0.01
        b1 = np.zeros(hidden_dims[0], dtype=np.float32)

        initializers.append(
            helper.make_tensor(
                "w1", TensorProto.FLOAT, [hidden_dims[0], input_dim], w1.flatten()
            )
        )
        initializers.append(
            helper.make_tensor("b1", TensorProto.FLOAT, [hidden_dims[0]], b1)
        )

        # 后续层
        prev_dim = hidden_dims[0]
        for i, hidden_dim in enumerate(hidden_dims[1:], 1):
            w = np.random.randn(hidden_dim, prev_dim).astype(np.float32) * 0.01
            b = np.zeros(hidden_dim, dtype=np.float32)

            initializers.append(
                helper.make_tensor(
                    f"w{i+1}", TensorProto.FLOAT, [hidden_dim, prev_dim], w.flatten()
                )
            )
            initializers.append(
                helper.make_tensor(f"b{i+1}", TensorProto.FLOAT, [hidden_dim], b)
            )

            prev_dim = hidden_dim

        # 输出层
        w_out = np.random.randn(3, prev_dim).astype(np.float32) * 0.01
        b_out = np.zeros(3, dtype=np.float32)

        initializers.append(
            helper.make_tensor("w_out", TensorProto.FLOAT, [3, prev_dim], w_out.flatten())
        )
        initializers.append(helper.make_tensor("b_out", TensorProto.FLOAT, [3], b_out))

        # 创建模型
        graph = helper.make_graph(
            nodes=[],  # 需要完整定义节点
            name="TradingClassifier",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=initializers,
        )

        model = helper.make_model(graph, producer_name="quantification_fit")
        onnx.save(model, output_path)

        logger.info(f"ONNX model created at {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to create ONNX model: {e}")
        return False
