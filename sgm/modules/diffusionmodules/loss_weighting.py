from abc import ABC, abstractmethod
from seva.geometry import get_camera_dist
import torch


class DiffusionLossWeighting(ABC):
    @abstractmethod
    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        pass


class UnitWeighting(DiffusionLossWeighting):
    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(sigma, device=sigma.device)


class EDMWeighting(DiffusionLossWeighting):
    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2


class VWeighting(EDMWeighting):
    def __init__(self):
        super().__init__(sigma_data=1.0)


class EpsWeighting(DiffusionLossWeighting):
    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma**-2.0

class EpsWeighting_v8:
    def __call__(self, sigma):
        weight = sigma**-2.0
        weight[weight<1]=1
        return weight

class EpsWeighting_v9:
    def __call__(self, sigma):
        weight = sigma**-2.0
        weight[weight<1]=1
        return weight

class EpsWeighting_v10:
    def __call__(self, sigma):
        weight = sigma**-2.0
        weight[weight<1]=sigma[weight<1]**-1.0
        return weight
    
class EpsWeighting_v11:
    def __call__(self, sigma):
        weight = sigma**-2.0
        weight[weight<1]=1
        return weight
    
class SevaWeighting:
    def __call__(self, c2ws: torch.Tensor,input_frame_mask: torch.Tensor,  num_frames: int,  max_weight: float = 5.0) -> torch.Tensor:
        """
        Args:
            c2ws: (B*T, 4, 4) camera to world matrices
            input_frame_mask: (B*T) boolean mask indicating condition images
            num_frames: int, number of frames in the batch
            max_weight: maximum weight value (default: 5.0)
        Returns:
            weights: (B*T) tensor of weights
        """
        c2ws = c2ws.view(-1, num_frames, 4, 4)
        input_frame_mask = input_frame_mask.view(-1, num_frames)
        B = c2ws.shape[0]
        # 计算所有c2ws到每个condition image的距离
        # 这样得到的是(B, T, sum(input_frame_mask[b]))，即每个batch中每个view到所有condition images的距离
        weights_list = []
        for b in range(B):
            # 计算当前batch中每个view到所有condition images的距离
            batch_distances = get_camera_dist(
                c2ws[b], 
                c2ws[b, input_frame_mask[b]], 
                mode="translation"
            ).min(-1).values  # (T)
            weights = batch_distances / batch_distances.max() * max_weight
            weights_list.append(weights)
        
        return torch.stack(weights_list, dim=0).flatten().to(c2ws.device)

# 测试代码
def test_seva_weighting():
    # 创建测试数据
    B, T = 2, 4
    c2ws = torch.randn(B*T, 4, 4)
    input_frame_mask = torch.zeros(B*T, dtype=torch.bool)
    input_frame_mask[0::T] = True  # 假设每个batch的第一帧是condition image
    
    # 创建权重计算器
    weighting = SevaWeighting()
    
    # 计算权重
    weights = weighting(c2ws, input_frame_mask, T)
    
    # 打印结果
    print("Input c2ws shape:", c2ws.shape)
    print("Input mask shape:", input_frame_mask.shape)
    print("Output weights shape:", weights.shape)
    print("\nWeights for each batch:")
    for b in range(B):
        print(f"Batch {b}:", weights[b])
    
    # 验证权重范围
    assert weights.min() >= 0.0, "Weights should be non-negative"
    assert weights.max() <= 5.0, "Weights should not exceed max_weight"
    
    # 验证每个batch的权重分布
    for b in range(B):
        # 检查最小距离的权重是否为0
        min_idx = torch.argmin(weights[b])
        assert weights[b, min_idx] == 0.0, "Minimum distance should have weight 0"
        
        # 检查最大距离的权重是否为max_weight
        max_idx = torch.argmax(weights[b])
        assert weights[b, max_idx] == 5.0, "Maximum distance should have weight max_weight"
        
        # 检查权重是否按距离递增
        sorted_weights = torch.sort(weights[b])[0]
        assert torch.all(torch.diff(sorted_weights) >= 0), "Weights should be monotonically increasing"

if __name__ == "__main__":
    test_seva_weighting()

