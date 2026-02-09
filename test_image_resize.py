#!/usr/bin/env python3
"""测试ImageResizePlus节点"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # 模拟ComfyUI环境
    class MockComfyUtils:
        @staticmethod
        def common_upscale(samples, width, height, upscale_method, crop):
            # 模拟common_upscale函数
            print(f"模拟common_upscale: width={width}, height={height}, method={upscale_method}, crop={crop}")
            return samples
    
    # 创建模拟模块
    import types
    mock_node_helpers = types.ModuleType('node_helpers')
    mock_comfy_utils = types.ModuleType('comfy.utils')
    mock_comfy_utils.common_upscale = MockComfyUtils.common_upscale
    
    sys.modules['node_helpers'] = mock_node_helpers
    sys.modules['comfy.utils'] = mock_comfy_utils
    
    # 现在导入节点
    from nodes import ImageResizePlus
    
    print("ImageResizePlus节点导入成功!")
    print(f"节点名称: {ImageResizePlus.__name__}")
    print(f"节点类别: {ImageResizePlus.CATEGORY}")
    
    # 检查INPUT_TYPES
    input_types = ImageResizePlus.INPUT_TYPES()
    print(f"输入类型: {input_types}")
    
    # 检查RETURN_TYPES
    print(f"返回类型: {ImageResizePlus.RETURN_TYPES}")
    print(f"返回名称: {ImageResizePlus.RETURN_NAMES}")
    
    print("\n测试完成!")
    
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()
