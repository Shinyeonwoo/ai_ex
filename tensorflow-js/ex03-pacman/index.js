/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import * as tfd from '@tensorflow/tfjs-data';

import {ControllerDataset} from './controller_dataset';
import * as ui from './ui';


// 예측하려는 클래스 수입니다. 이 예에서 우리는
// 위, 아래, 왼쪽, 오른쪽에 대한 4개의 클래스를 예측합니다.
const NUM_CLASSES = 4;

// 웹캠의 이미지에서 Tensor를 생성하는 웹캠 반복자입니다.
let webcam;

// 활성화를 저장할 데이터세트 개체입니다.
const controllerDataset = new ControllerDataset(NUM_CLASSES);

let truncatedMobileNet;
let model;

// mobilenet을 로드하고 내부 활성화를 반환하는 모델을 반환합니다.
// 분류기 모델에 대한 입력으로 사용합니다.
async function loadTruncatedMobileNet() {
  const mobilenet = await tf.loadLayersModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // 내부 활성화를 출력하는 모델을 반환합니다.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

// UI 버튼을 누르면 웹캠에서 프레임을 읽고 연결합니다.
// 버튼에 지정된 클래스 라벨을 사용합니다. 위, 아래, 왼쪽, 오른쪽은
// 각각 0, 1, 2, 3에 라벨을 붙입니다.
ui.setExampleHandler(async label => {
  let img = await getImage();

  controllerDataset.addExample(truncatedMobileNet.predict(img), label);

  // 미리보기 썸네일을 그립니다.
  ui.drawThumb(img, label);
  img.dispose();
})

/**
 * 분류기를 설정하고 훈련시킵니다.
 */
async function train() {
  if (controllerDataset.xs == null) {
    throw new Error('Add some examples before training!');
  }

  // 2계층 완전 연결 사진을 생성합니다. 별도의 사진을 생성하여,
  // mobilenet 모델에 레이어를 추가하는 대신에 "동결"합니다.
  // mobilenet 모델의 모델이고 새 모델의 사용자만 훈련합니다.
  model = tf.sequential({
    layers: [
      // 입력을 벡터로 평면화하여 밀집 레이어에서 사용할 수 있습니다. 하는 동안
      // 기술적으로는 레이어이며 모양 변경만 수행합니다. (교육은 없습니다.)
      // 매개변수).
      tf.layers.flatten(
          {inputShape: truncatedMobileNet.outputs[0].shape.slice(1)}),
      // Layer 1.
      tf.layers.dense({
        units: ui.getDenseUnits(),
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // 레이어 2. 마지막 레이어의 단위 수는 일치해야 합니다.
      // 예측하려는 클래스 수입니다.
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  // 모델 훈련을 구동하는 최적화 프로그램을 생성합니다.
  const optimizer = tf.train.adam(ui.getLearningRate());
  // 우리는 손실 함수인 categoricalCrossentropy를 사용합니다.
  // 예측된 항목 간의 오류를 측정하는 범주형 분류
  // 클래스에 대한 확률 분포(입력이 각 클래스에 속할 확률)
  // 클래스), 라벨(실제 클래스에서 100% 확률)>
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

  // 배치 크기를 전체 데이터세트의 일부로 매개변수화합니다.
  // 수집되는 예시 수는 사용자가 얼마나 많은 예시를 사용하는지에 따라 달라집니다.
  // 수집합니다. 이를 통해 유연한 배치 크기를 가질 수 있습니다.
  const batchSize =
      Math.floor(controllerDataset.xs.shape[0] * ui.getBatchSizeFraction());
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  // 모델을 훈련시킵니다! Model.fit()은 xs와 ys를 섞으므로 그럴 필요가 없습니다.
  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: ui.getEpochs(),
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        ui.trainStatus('Loss: ' + logs.loss.toFixed(5));
      }
    }
  });
}

let isPredicting = false;

async function predict() {
  ui.isPredicting();
  while (isPredicting) {
    
    // 웹캠에서 프레임을 캡처합니다.
    const img = await getImage();

    // 모바일넷을 통해 예측하고 내부 활성화를 얻습니다.
    // 모바일넷 모델, 즉 입력 이미지의 "임베딩"입니다.
    const embeddings = truncatedMobileNet.predict(img);

    // 임베딩을 사용하여 새로 학습된 모델을 통해 예측합니다.
    // mobilenet에서 입력으로.
    const predictions = model.predict(embeddings);

    // 최대 확률의 인덱스를 반환합니다. 이 숫자는 해당
    // 모델이 입력에 대해 가장 가능성이 있다고 생각하는 클래스입니다.
    const predictedClass = predictions.as1D().argMax();
    const classId = (await predictedClass.data())[0];
    img.dispose();

    ui.predictClass(classId);
    await tf.nextFrame();
  }
  ui.donePredicting();
}

/**
 * 웹캠에서 프레임을 캡처하여 -1과 1 사이에서 정규화합니다.
 * [1, w, h, c] 모양의 배치 이미지(요소 1개 배치)를 반환합니다.
 */
async function getImage() {
  const img = await webcam.capture();
  const processedImg =
      tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
  img.dispose();
  return processedImg;
}

document.getElementById('train').addEventListener('click', async () => {
  ui.trainStatus('Training...');
  await tf.nextFrame();
  await tf.nextFrame();
  isPredicting = false;
  train();
});
document.getElementById('predict').addEventListener('click', () => {
  ui.startPacman();
  isPredicting = true;
  predict();
});

async function init() {
  try {
    webcam = await tfd.webcam(document.getElementById('webcam'));
  } catch (e) {
    console.log(e);
    document.getElementById('no-webcam').style.display = 'block';
  }
  truncatedMobileNet = await loadTruncatedMobileNet();

  ui.init();

  
  // 모델을 워밍업합니다. 이는 GPU에 가중치를 업로드하고 WebGL을 컴파일합니다.
  // 프로그래밍하므로 웹캠에서 처음으로 데이터를 수집할 때
  // 빠른.
  const screenShot = await webcam.capture();
  truncatedMobileNet.predict(screenShot.expandDims(0));
  screenShot.dispose();
}

// 애플리케이션을 초기화합니다.
init();