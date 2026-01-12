# Graphviz 설정 및 모델 시각화 가이드

## 문제 해결 완료! ✅

`visualize_model.py` 파일이 이미 수정되어 Graphviz PATH를 자동으로 추가합니다.

## 실행 방법

### 방법 1: 직접 Python 실행 (권장)

새 터미널(또는 명령 프롬프트)을 열고 다음 명령어를 실행하세요:

```bash
cd C:\Users\hmc2020\Desktop\kari-onestop-uas\physics_network
python visualize_model.py
```

### 방법 2: 배치 파일 사용

`run_visualize.bat` 파일을 더블클릭하거나 다음 명령어를 실행하세요:

```bash
cd C:\Users\hmc2020\Desktop\kari-onestop-uas\physics_network
run_visualize.bat
```

## 변경 사항

`visualize_model.py` 파일에 다음 코드가 추가되었습니다:

```python
# Graphviz PATH 추가 (Windows)
graphviz_bin = r"C:\Program Files\Graphviz\bin"
if os.path.exists(graphviz_bin):
    os.environ["PATH"] += os.pathsep + graphviz_bin
```

이제 스크립트가 실행될 때 자동으로 Graphviz를 찾을 수 있습니다.

## 출력 파일

성공적으로 실행되면 다음 파일들이 생성됩니다:

1. **터미널 출력:**
   - torchinfo summary (레이어별 상세 정보)
   - ASCII 아키텍처 다이어그램
   - Layer-by-layer breakdown
   - Parameter statistics

2. **이미지 파일:**
   - `model_graph.png` - Computational graph (torchviz 사용)

## 참고 사항

- 학습 중에도 실행 가능합니다 (독립적인 프로세스)
- GPU 메모리를 사용하지 않습니다 (CPU only)
- 처음 실행 시 torchviz/torchinfo 설치가 필요할 수 있습니다:
  ```bash
  pip install torchinfo torchviz
  ```

## 시스템 PATH 영구 추가 (선택사항)

매번 스크립트에서 PATH를 설정하고 싶지 않다면, 시스템 환경 변수에 영구적으로 추가하세요:

1. "시스템 환경 변수 편집" 검색
2. "환경 변수" 버튼 클릭
3. "시스템 변수" 섹션에서 "Path" 선택 → "편집"
4. "새로 만들기" 클릭
5. `C:\Program Files\Graphviz\bin` 입력
6. 확인 → 확인 → 확인
7. **중요:** 모든 터미널/IDE 재시작

## 확인 방법

Graphviz가 제대로 설치되었는지 확인하려면:

```bash
dot -V
```

출력 예시:
```
dot - graphviz version 2.50.0 (...)
```

## 문제가 계속된다면

1. Python 및 필요한 패키지 설치 확인:
   ```bash
   pip list | grep -i torch
   pip list | grep -i graphviz
   ```

2. Graphviz 실행 파일 존재 확인:
   ```bash
   dir "C:\Program Files\Graphviz\bin\dot.exe"
   ```

3. 스크립트 수정 내역 확인:
   ```bash
   head -n 20 visualize_model.py
   ```
   처음 몇 줄에 Graphviz PATH 추가 코드가 있어야 합니다.
