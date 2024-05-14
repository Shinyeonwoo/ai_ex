import {MnistData} from './data.js';



async function showExamples(data) {
  // Create a container in the visor
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];

  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });

    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

async function run() {
  const data = new MnistData();
  await data.load();
  await showExamples(data);

  // 모델 아키텍처 정의 및 모델 학습
  const model = getModel();
  tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model);

  await train(model, data);
}

document.addEventListener('DOMContentLoaded', run);


function getModel() {
  const model = tf.sequential();

  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;

  // In the first layer of our convolutional neural network we have
  // to specify the input shape. Then we specify some parameters for
  // the convolution operation that takes place in this layer.
  /**
   * 컨볼루션
   * 여기서는 순차적 모델을 사용한다. 밀집 레이어 대신 con2d 레이어를 사용한다
   */
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS], // 모델의 첫 번째 레이어로 전달될 데이터의 모양임. 이 경우 28x28의 흑백 이미지 자세한 내용은 설명서 참조
    kernelSize: 5, // 입력 데이터에 적용되는 슬라이딩 컨볼루셔널 필터 창
    filters: 8, // 입력 데이터의 적용할 kernelSize 크기의 필터 창, 여기선 8개의 필터 적용
    strides: 1, // 슬라이딩 창의 '보폭' 즉 이미지에서 이동할 때마다 필터가 변경하는 픽셀 수 , 여기서는 1픽셀씩 이동
    activation: 'relu', // 컨볼루션이 완료된 후 데이터에 적용할 활성화 함수임 이 경우에는 ML모델에서 일반적인 활성화 함수인 정류 선형 유닛(ReLU)를 사용
    kernelInitializer: 'varianceScaling'  // 모델 가중치를 무작위로 초기화하는 데 사용하는 메서드로 동적인 학습에 매우 중요.여기서는 일반적으로 유용한 옵션 사용
  }));

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.
  /**
   * 데이터 표현 평면화
   * 이미지는 고차원 데이터이며 컨볼루션 연산에서는 전달된 데이터의 크기를 늘리는 경향이 있다.
   * 최종 분류 레이어로 전달하기 전에 데이터를 하나의 긴 배열로 평면화 해야한다.
   * 최종 레이어로 사용하는 밀집 레이어에서 tensor1d만 사용하므로 이 단계는 대부분의 분류 작업에서 일반적으로 사용된다.
   * 참고로 평면화된 레이어에는 가중치가 없다. 단지 입력을 긴 배열로 전개할 뿐임
   */
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  // Repeat another conv2d + maxPooling stack.
  // Note that we have more filters in the convolution.
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten());

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  /**
   * 최종 확률 분포 계산
   * 밀집 레이어와 소프트맥스 활성화를 함께 사용하여 가능한 10개의 클래스에 대한 확률 분포를 계산한다. 점수가 가장 높은 클래스가 예측 숫자가 됨
   */
  const NUM_OUTPUT_CLASSES = 10;
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));

  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  /**
   * 옵티마이저 및 손실 합수 선택
   * categoricalCrossentropy는 모델의 출력이 확률 분포일 때 사용된다. (ex1 예제와 다름)
   * 모델의 마지막 레이어에서 생성된 확률 분포와 true라벨에서 지정한 확률 분포 간의 오차를 측정함
   */
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

async function train(model, data) {
  /**
   * 측정항목 모니터링
   * 여기서는 모니터링할 측정항목을 결정한다. 학습 세트에서 손실과 정확성을 모니터링하고 검증 세트에서의 손실과 정확성도 각각 모니터링함.
   */
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = {
    name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  /**
   * 데이터를 텐서로 준비
   * 여기서는 2개의 데이터 세트, 즉 모델을 학습시킬 학습 세트와 각 세대가 끝날 때 모델을 테스트할 검증 세트를 만듭니다.
   * 단, 검증 세트의 데이터는 학습 중에 모델에 표시되지 않습니다.
   * 
   * 제공된 데이터 클래스를 사용하면 이미지 데이터에서 텐서를 쉽게 가져올 수 있습니다.
   * 하지만 모델에 전달하기 전에 모델에서 예상하는 모양 ([num_examples, image_width, image_height, channels])으로 텐서를 바꿔야 합니다.
   * 각 데이터 세트에는 입력(X)과 라벨(Y)이 모두 있습니다.
   * 
   * 참고로 trainDataSize를 5,500으로 설정하고 testDataSize를 1,000으로 설정하면 빠르게 실험을 진행 할 수 있습니다.
   */
  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [
      d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [
      d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  /**
   * model.fit을 호출해 학습 루프를 시작합니다.
   * 또한 각 세대가 끝난 후 모델이 테스트에 사용할 데이터(학습에는 사용하지 않음)를 나타내기 위해 validationData 속성도 전달합니다.
   * 
   * 학습 데이터는 결과가 좋지만 검증 데이터는 결과가 좋지 않다면 모델이 학습 데이터에 과적합했을 가능성이 크며 이전에 인식되지 않은 입력에는 효과적으로 일반화되지 않는다는 의미입니다.
   */
  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks
  });
}


