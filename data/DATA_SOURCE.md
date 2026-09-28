# Data Source & Description

본 프로젝트에서 사용하는 데이터셋은 캐글(Kaggle)에 공개된 **HAM10000** 피부암/피부 병변 이미지 데이터셋입니다.

## 1. 데이터셋 개요

- **데이터셋 이름**: HAM10000 ("Human Against Machine with 10000 training images")
- **출처**: [Kaggle - Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **총 데이터 규모**: 10,015건의 다중 소스 피부경검사(Dermatoscopic) 이미지

## 2. 주요 클래스 (7대 피부 질환)

데이터셋은 다음과 같은 7가지 주요 피부 병변 클래스로 구성되어 있습니다[cite: 24]:

1. **Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)**
2. **Basal cell carcinoma (bcc)**
3. **Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses) (bkl)**
4. **Dermatofibroma (df)**
5. **Melanocytic nevi (nv)**
6. **Pyogenic granulomas and hemorrhage (vasc)**
7. **Melanoma (mel)**

## 3. 데이터 전처리 및 활용 노트

- **결측치 처리**: 연령(age) 등의 메타데이터 결측치는 평균값으로 대체 및 정수형 변환을 거쳐 정제되었습니다[cite: 24].
- **이미지 규격**: 모델 학습 효율을 위해 28x28 픽셀 규격으로 표준화되었습니다.
- **클래스 불균형 해소**: 데이터 증강(Augmentation) 기법을 적용하여 총 45,756건의 확장된 학습 데이터를 확보하여 활용했습니다[cite: 25].
