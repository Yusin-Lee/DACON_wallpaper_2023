# Dacon_wallpaper_2023
2023년 데이콘 도배 하자 유형 분류 대회
https://dacon.io/competitions/official/236082/overview/description

1. Augmentation
- 데이터의 라벨링이 잘못된 경우가 존재하여 학습 데이터에 대해 라벨링을 조사하여 다시 라벨링을 하였음.
- 데이터 라벨 별 각도 조정에 관계 없는 데이터에 대해 90도 조정한 데이터를 추가
- 좌우 혹은 상하에 상관 없는 데이터 중 데이터 개수가 적은 경우 replicate
- 모든 데이터에 대해 p = 0.5의 확률로 RandomBrightnessContrast 적용

2. 실험 과정
- 가장 기초 데이터 셋( + 좌우, 상하 augmentation) + EfficientNet_b0_ns ~ EfficientNet_b7_ns 까지 학습한 결과 약 0.45 ~ 0.55 수준의 public score가 기록됨
- relabeling( + 좌우, 상하) + EfficientNet, ViT로 학습한 결과 0.62 수준의 public score가 기록됨
- relabeling( + 좌우, 상하, Brightness Augmentation) + ConvNext-large로 학습한 결과 0.68 수준의 public score가 기록됨 -> 최종 private score : 0.70

3. 결과
- 기존의 데이터 셋에 소수 데이터 중 일부는 라벨링이 잘못된 경우가 존재, 그러므로 이 라벨에 대해서 학습이 잘 안되는 경우가 발생
- 라벨링 및 augmentation 후 전체적으로 소수의 데이터에 대해서도 학습이 어느정도 진행되는 것을 확인
