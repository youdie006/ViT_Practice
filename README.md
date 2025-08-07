# Vision Transformer (ViT) Implementation for CIFAR-10

## 프로젝트 개요
Vision Transformer(ViT)를 처음부터 구현하여 CIFAR-10 데이터셋에서 이미지 분류를 수행하는 프로젝트

## 주요 구성 요소

### 1. Patch Projection
- 이미지를 패치로 분할
- Conv2d를 사용한 linear projection
- 입력: (batch, 3, 224, 224) → 출력: (batch, 196, 768)

### 2. Patch Embedding
- CLS 토큰 추가
- Positional encoding 적용
- 입력: (batch, 3, 224, 224) → 출력: (batch, 197, 768)

### 3. Multi-Head Attention (MHA)
- 12개의 attention heads
- Scaled dot-product attention
- QKV projection 및 output projection

### 4. Transformer Encoder Block
- Pre-normalization with LayerNorm
- MHA + Residual connection
- MLP (4x hidden dimension) + Residual connection

### 5. Vision Transformer
- 12개의 Transformer blocks
- Classification head with CLS token

## 학습 결과

### 성능
- **최종 테스트 정확도**: 69.08%
- **학습 시간**: 약 2.5시간 (30 epochs, MPS)
- **모델 크기**: 4.77M parameters

### 학습 설정
```python
- Optimizer: AdamW (lr=5e-4, weight_decay=0.03)
- Scheduler: Cosine annealing with warmup
- Batch size: 256
- Label smoothing: 0.1
```

## 예측 결과
![ViT Predictions](vit_predictions.png)

## 모델 구성 비교

| Configuration | Embedding Size | Depth | Heads | Parameters |
|--------------|---------------|-------|--------|------------|
| ViT-Tiny     | 192          | 12    | 3      | 5.7M       |
| ViT-Small    | 384          | 12    | 6      | 22M        |
| ViT-Base     | 768          | 12    | 12     | 86.5M      |
| ViT-Large    | 1024         | 24    | 16     | 304M       |

## CIFAR-10 적용 변경사항
- 이미지 크기: 32x32 (원본 224x224 대신)
- 패치 크기: 4x4 (원본 16x16 대신)
- 총 패치 수: 64개

## 참고 자료
- [An Image is Worth 16x16 Words (논문)](https://arxiv.org/abs/2010.11929)
- [Vision Transformer (Google Research)](https://github.com/google-research/vision_transformer)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)