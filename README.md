# Pix2Pix


Pix2PixGAN은 이미지를 이미지로 Generator을 학습시키고, fake이미지를 출력하고, 그 이미지를 discriminator가 완성된 그림을 식별하도록 목적 함수를 설계하여, 학습 시키면서 서서히 generator은 완성된 fake그림을 출력하여, ground truth에 가까운 그림을 만들어 discriminator가 구별하지 못하도록 한다,

논문을 통해, UNet과 PacthGAN모델로 구성했을 때, 높은 성능의 결과로 보인다고하여, Pix2PixGAN을 구성할 때 활용했다. Train과 Test를 조건대로 8:2로 나누어 테스트했다.
