import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        # 1. 创建 pos 向量: shape (seq_len, 1)
        pos = np.arange(seq_len)[:, np.newaxis]
        
        # 2. 创建 i 向量 (对应公式中的 2i): shape (d_model // 2,)
        # 因为正弦和余弦共用相同的分母部分，我们只计算偶数索引部分
        i = np.arange(0, d_model, 2)
        
        # 3. 计算分母部分: 10000^(2i / d_model)
        # 利用指数对数变换优化计算 (可选): exp(2i * -log(10000) / d_model)
        div_term = 10000 ** (i / d_model)
        
        # 4. 初始化编码矩阵
        pe = np.zeros((seq_len, d_model))
        
        # 5. 广播计算并赋值
        # PE(pos, 2i) = sin(...)
        pe[:, 0::2] = np.sin(pos / div_term)
        # PE(pos, 2i+1) = cos(...)
        pe[:, 1::2] = np.cos(pos / div_term)
        
        return np.round(pe, 5)
    
