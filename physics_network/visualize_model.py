"""
Model Architecture Visualization Script
모델 구조를 텍스트와 이미지로 시각화
"""
import os
import sys

# Graphviz PATH 추가 (Windows)
graphviz_bin = r"C:\Program Files\Graphviz\bin"
if os.path.exists(graphviz_bin):
    os.environ["PATH"] += os.pathsep + graphviz_bin

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import torch#; import torchinfo; import torchviz
from model.model_seq2seq_v2 import ConcentrationSeq2Seq_v2

# ============================================
# Method 1: torchinfo (가장 추천!)
# ============================================
def visualize_with_torchinfo():
    """torchinfo를 사용한 상세한 구조 출력"""
    try:
        from torchinfo import summary

        print("=" * 80)
        print("📊 Model Architecture Summary (torchinfo)")
        print("=" * 80)

        # 모델 생성
        model = ConcentrationSeq2Seq_v2(
            hidden_channels=32,
            num_lstm_layers=2,
            output_shape=(21, 45, 45)
        )

        # 입력 shape 정의
        batch_size = 8
        seq_len = 30

        # Summary 출력
        summary(
            model,
            input_size=[
                (batch_size, seq_len, 21, 45, 45),  # past_conc
                (batch_size, 2, 21, 45, 45)         # static_maps
            ],
            col_names=["input_size", "output_size", "num_params", "kernel_size"],
            depth=5,
            device="cpu"
        )

        print("\n✅ torchinfo summary completed!")
        return True

    except ImportError:
        print("⚠️ torchinfo not installed. Install with: pip install torchinfo")
        return False


# ============================================
# Method 2: torchviz (그래프 이미지 생성)
# ============================================
def visualize_with_torchviz():
    """torchviz를 사용한 computational graph 생성"""
    try:
        from torchviz import make_dot

        print("\n" + "=" * 80)
        print("🎨 Generating Computational Graph (torchviz)")
        print("=" * 80)

        # 모델 생성
        model = ConcentrationSeq2Seq_v2(
            hidden_channels=32,
            num_lstm_layers=2,
            output_shape=(21, 45, 45)
        )

        # 작은 입력으로 forward pass
        batch_size = 2
        past_conc = torch.randn(batch_size, 30, 21, 45, 45)
        static_maps = torch.randn(batch_size, 2, 21, 45, 45)

        # Forward
        output = model(past_conc, static_maps)

        # Computational graph 생성
        dot = make_dot(
            output,
            params=dict(model.named_parameters()),
            show_attrs=True,
            show_saved=True
        )

        # 저장
        output_path = "model_graph"
        dot.render(output_path, format='png', cleanup=True)

        print(f"✅ Computational graph saved to: {output_path}.png")
        return True

    except ImportError:
        print("⚠️ torchviz not installed. Install with: pip install torchviz graphviz")
        print("   Also install Graphviz: https://graphviz.org/download/")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


# ============================================
# Method 3: Custom ASCII Visualization
# ============================================
def visualize_custom_ascii():
    """커스텀 ASCII 아트로 구조 시각화"""
    print("\n" + "=" * 80)
    print("📐 Custom Model Architecture Diagram")
    print("=" * 80)

    diagram = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    ConcentrationSeq2Seq_v2 Architecture                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ INPUT LAYER                                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  past_conc:    (B=8, T=30, D=21, H=45, W=45)  ─────┐                       │
│                                                     │                       │
│  static_maps:  (B=8, C=2,  D=21, H=45, W=45)  ──┐  │                       │
│                                                  │  │                       │
└──────────────────────────────────────────────────┼──┼───────────────────────┘
                                                   │  │
                ┌──────────────────────────────────┘  │
                │                                     │
                ▼                                     ▼
    ┌───────────────────────┐          ┌────────────────────────┐
    │  StaticEncoder        │          │  ConvLSTMEncoder       │
    ├───────────────────────┤          ├────────────────────────┤
    │                       │          │                        │
    │  Conv3d(2→16)         │          │  input_conv:           │
    │  BatchNorm + ReLU     │          │    Conv3d(1→32)        │
    │                       │          │    BatchNorm + ReLU    │
    │  Conv3d(16→32)        │          │                        │
    │  BatchNorm + ReLU     │          │  ConvLSTM Layer 1:     │
    │                       │          │    ┌─────────────┐    │
    │                       │          │    │  for t=0..29│    │
    │                       │          │    │  ┌────────┐ │    │
    │                       │          │    │  │ h_t, c │ │    │
    │                       │          │    │  │  LSTM  │ │    │
    │                       │          │    │  └────────┘ │    │
    │                       │          │    └─────────────┘    │
    │                       │          │                        │
    │                       │          │  ConvLSTM Layer 2:     │
    │                       │          │    (same structure)    │
    │                       │          │                        │
    └───────────────────────┘          └────────────────────────┘
                │                                   │
                │  (B, 32, 21, 45, 45)              │  (B, 32, 21, 45, 45)
                │                                   │
                │                                   │
                └────────────┬──────────────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │     UNetDecoder            │
                ├────────────────────────────┤
                │                            │
                │  Fusion Layer:             │
                │    concat → (B, 64, ...)   │
                │    Conv3d(64→32)           │
                │    BatchNorm + ReLU        │
                │                            │
                │  Decoder:                  │
                │    Conv3d(32→32)           │
                │    BatchNorm + ReLU        │
                │                            │
                │    Conv3d(32→16)           │
                │    BatchNorm + ReLU        │
                │                            │
                │    Conv3d(16→1)            │
                │                            │
                │  softplus (non-negative)   │
                │                            │
                └────────────────────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │  OUTPUT                    │
                ├────────────────────────────┤
                │  pred_conc:                │
                │  (B=8, 1, D=21, H=45, W=45)│
                └────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════════════╗
║  Key Components:                                                               ║
║                                                                                ║
║  1. ConvLSTMEncoder: Processes 30 timesteps sequentially                      ║
║     - Maintains spatial structure (21×45×45) throughout                       ║
║     - LSTM hidden state captures temporal dependencies                        ║
║                                                                                ║
║  2. StaticEncoder: Encodes terrain and emission sources                       ║
║     - Simple 2-layer Conv3d                                                   ║
║                                                                                ║
║  3. UNetDecoder: Fuses temporal + static features                             ║
║     - Concatenates both feature maps                                          ║
║     - Decodes to final prediction                                             ║
║                                                                                ║
║  Total Parameters: ~402,849                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

    print(diagram)

    # 파라미터 수 계산
    model = ConcentrationSeq2Seq_v2(
        hidden_channels=32,
        num_lstm_layers=2,
        output_shape=(21, 45, 45)
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 80)
    print("📊 Parameter Statistics")
    print("=" * 80)
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"Model size (approx):   {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    # 각 컴포넌트별 파라미터 수
    print("\n" + "-" * 80)
    print("Component-wise Parameters:")
    print("-" * 80)

    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{name:30s}: {num_params:>10,} parameters")


# ============================================
# Method 4: Layer-by-layer breakdown
# ============================================
def visualize_layer_details():
    """레이어별 상세 정보 출력"""
    print("\n" + "=" * 80)
    print("🔍 Layer-by-Layer Breakdown")
    print("=" * 80)

    model = ConcentrationSeq2Seq_v2(
        hidden_channels=32,
        num_lstm_layers=2,
        output_shape=(21, 45, 45)
    )

    print("\n1️⃣  ConvLSTMEncoder (temporal_encoder)")
    print("-" * 80)
    for name, layer in model.temporal_encoder.named_modules():
        if isinstance(layer, (torch.nn.Conv3d, torch.nn.BatchNorm3d)):
            print(f"  {name:40s}: {layer}")

    print("\n2️⃣  StaticEncoder (static_encoder)")
    print("-" * 80)
    for name, layer in model.static_encoder.named_modules():
        if isinstance(layer, (torch.nn.Conv3d, torch.nn.BatchNorm3d)):
            print(f"  {name:40s}: {layer}")

    print("\n3️⃣  UNetDecoder (decoder)")
    print("-" * 80)
    for name, layer in model.decoder.named_modules():
        if isinstance(layer, (torch.nn.Conv3d, torch.nn.BatchNorm3d)):
            print(f"  {name:40s}: {layer}")


# ============================================
# Main execution
# ============================================
def main():
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "MODEL VISUALIZATION SUITE" + " " * 33 + "║")
    print("╚" + "═" * 78 + "╝\n")

    # Method 1: torchinfo (권장)
    success_torchinfo = visualize_with_torchinfo()

    # Method 2: Custom ASCII
    visualize_custom_ascii()

    # Method 3: Layer details
    visualize_layer_details()

    # Method 4: torchviz (선택적)
    print("\n" + "=" * 80)
    print("Would you like to generate computational graph image? (requires graphviz)")
    print("This will create a PNG file showing the forward pass.")
    print("=" * 80)

    # 자동으로 시도
    visualize_with_torchviz()

    print("\n" + "=" * 80)
    print("✅ Visualization Complete!")
    print("=" * 80)

    if not success_torchinfo:
        print("\n💡 Tip: Install torchinfo for detailed layer information:")
        print("   pip install torchinfo")

    print("\n💡 Tip: For computational graph (PNG image):")
    print("   1. Install: pip install torchviz")
    print("   2. Install Graphviz: https://graphviz.org/download/")
    print("   3. Add Graphviz to PATH")


if __name__ == "__main__":
    main()
