/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * ...
 */

import * as tf from '@tensorflow/tfjs'; // TensorFlow.js 라이브러리 임포트
import * as tfd from '@tensorflow/tfjs-data'; // 데이터 처리용 TensorFlow.js 데이터 모듈 임포트

import {ControllerDataset} from './controller_dataset'; // 학습 데이터셋을 관리하는 클래스 임포트
import * as ui from './ui'; // 사용자 인터페이스 관련 기능 임포트

// 예측하려는 클래스 수입니다. 여기서는 위, 아래, 왼쪽, 오른쪽을 예측하므로 4개의 클래스가 필요합니다.
const NUM_CLASSES = 4;

// 웹캠에서 이미지에서 Tensor를 생성하는 반복자 변수입니다.
let webcam;

// 활성화를 저장할 데이터셋 객체입니다.
const controllerDataset = new ControllerDataset(NUM_CLASSES);

let truncatedMobileNet; // MobileNet 모델의 일부 계층만 사용하도록 조정한 모델
let model; // 새롭게 학습할 분류기 모델

// MobileNet을 로드하고 내부 활성화를 반환하는 모델을 반환합니다. 이 모델은 분류기의 입력으로 사용됩니다.
async function loadTruncatedMobileNet() {
  // 사전 훈련된 MobileNet 모델을 로드합니다.
  const mobilenet = await tf.loadLayersModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // MobileNet 모델에서 특정 레이어(conv_pw_13_relu)를 가져와 활성화를 추출할 수 있도록 합니다.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

// UI에서 버튼을 누르면 웹캠에서 프레임을 읽고 해당 클래스 레이블을 사용하여 연결합니다.
ui.setExampleHandler(async label => {
  let img = await getImage(); // 웹캠에서 이미지를 캡처합니다.

  // MobileNet의 활성화를 예측하고 이를 학습 데이터셋에 추가합니다.
  controllerDataset.addExample(truncatedMobileNet.predict(img), label);

  // 썸네일을 UI에 그려 사용자가 추가한 예시를 확인할 수 있도록 합니다.
  ui.drawThumb(img, label);
  img.dispose(); // 사용한 이미지 텐서를 메모리에서 해제합니다.
});

/**
 * 모델을 설정하고 훈련시킵니다.
 */
async function train() {
  if (controllerDataset.xs == null) {
    throw new Error('Add some examples before training!'); // 학습 전에 예제가 하나도 없으면 에러를 발생시킵니다.
  }

  // 분류 모델을 생성합니다. MobileNet의 출력을 입력으로 받아 분류를 수행하도록 레이어를 추가합니다.
  model = tf.sequential({
    layers: [
      // 입력을 벡터 형태로 평탄화합니다.
      tf.layers.flatten(
          {inputShape: truncatedMobileNet.outputs[0].shape.slice(1)}),
      // 첫 번째 은닉 레이어 (relu 활성화 함수 사용)
      tf.layers.dense({
        units: ui.getDenseUnits(),
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // 출력 레이어 (클래스 수 만큼의 출력을 생성, softmax 활성화 함수 사용)
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  // Adam 최적화 프로그램을 사용하여 모델을 훈련시킵니다.
  const optimizer = tf.train.adam(ui.getLearningRate());
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'}); // categoricalCrossentropy 손실 함수 사용

  // 배치 크기를 데이터셋의 일부로 매개변수화하여 설정합니다.
  const batchSize =
      Math.floor(controllerDataset.xs.shape[0] * ui.getBatchSizeFraction());
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`); // 배치 크기가 0 또는 NaN이면 오류 발생
  }

  // 모델을 훈련시킵니다.
  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: ui.getEpochs(),
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        ui.trainStatus('Loss: ' + logs.loss.toFixed(5)); // 배치가 끝날 때마다 현재 손실 값을 UI에 업데이트
      }
    }
  });
}

let isPredicting = false; // 예측 여부 상태 플래그

async function predict() {
  ui.isPredicting(); // 예측 중이라는 상태를 UI에 표시
  while (isPredicting) {
    const img = await getImage(); // 웹캠에서 프레임을 캡처합니다.

    // MobileNet을 사용하여 예측한 활성화를 가져옵니다.
    const embeddings = truncatedMobileNet.predict(img);

    // 새로 학습한 모델을 통해 예측합니다.
    const predictions = model.predict(embeddings);

    // 가장 높은 확률을 가진 클래스의 인덱스를 찾습니다.
    const predictedClass = predictions.as1D().argMax();
    const classId = (await predictedClass.data())[0];
    img.dispose(); // 이미지 텐서 해제

    ui.predictClass(classId); // 예측한 클래스 결과를 UI에 표시
    await tf.nextFrame(); // 다음 프레임을 위해 대기
  }
  ui.donePredicting(); // 예측이 끝났음을 UI에 알림
}

/**
 * 웹캠에서 이미지를 캡처하고 -1에서 1 사이로 정규화합니다.
 * [1, w, h, c] 형태의 배치 이미지를 반환합니다.
 */
async function getImage() {
  const img = await webcam.capture();
  const processedImg = tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1)); // 이미지 정규화
  img.dispose(); // 사용한 이미지 텐서 해제
  return processedImg;
}

// 훈련 버튼 클릭 시 모델 훈련 시작
document.getElementById('train').addEventListener('click', async () => {
  ui.trainStatus('Training...');
  await tf.nextFrame(); // 첫 번째 프레임 건너뛰기
  await tf.nextFrame(); // 두 번째 프레임 건너뛰기
  isPredicting = false; // 훈련 중이므로 예측 상태를 false로 설정
  train(); // 모델 훈련 시작
});

// 예측 버튼 클릭 시 예측 시작
document.getElementById('predict').addEventListener('click', () => {
  ui.startPacman(); // Pacman 게임 시작
  isPredicting = true; // 예측 상태로 설정
  predict(); // 예측 함수 호출
});

// 애플리케이션 초기화 함수
async function init() {
  try {
    webcam = await tfd.webcam(document.getElementById('webcam')); // 웹캠 초기화
  } catch (e) {
    console.log(e); // 웹캠 사용 불가 시 오류 로그 출력
    document.getElementById('no-webcam').style.display = 'block'; // 웹캠 없음을 표시하는 메시지 보여줌
  }
  truncatedMobileNet = await loadTruncatedMobileNet(); // MobileNet 모델 로드
  ui.init(); // UI 초기화

  // 모델을 워밍업하여 첫 사용 시 빠르게 반응할 수 있도록 합니다.
  const screenShot = await webcam.capture();
  truncatedMobileNet.predict(screenShot.expandDims(0));
  screenShot.dispose();
}

// 애플리케이션을 초기화합니다.
init();
