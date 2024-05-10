console.log('Hello TensorFlow');
/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
    const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData = await carsDataResponse.json();
    const cleaned = carsData.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
    }))
    .filter(car => (car.mpg != null && car.horsepower != null));

    return cleaned;
}

async function run() {
    // Load and plot the original input data that we are going to train on.
    const data = await getData();
    const values = data.map(d => ({
        x: d.horsepower,
        y: d.mpg,
    }));

    tfvis.render.scatterplot(
        {name: 'Horsepower v MPG'},
        {values},
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    );

    // 모델 인스턴스화
    const model = createModel();
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    // 모델 학습
    // Convert the data to a form we can use for training.
    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;

    // Train the model
    await trainModel(model, inputs, labels);
    console.log('Done Training');
}

document.addEventListener('DOMContentLoaded', run);

function createModel() {
    // 모델 인스턴스화
    const model = tf.sequential();

    // 레이어 최초 추가
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

    // 레이어 추가
    model.add(tf.layers.dense({units: 1, useBias: true}));

    return model;
}

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

    return tf.tidy(() => {
        // Step 1. 데이터 셔플링
        // 학습 알고리즘에 제공할 예시의 순서를 무작위로 지정
        // 일반적으로 학습하는 동안 데이터 세트는 모델이 학습할 크기가 작은 하위 집합(배치)으로 분할되기 대문에 셔플이 중요함
        // 셔플은 각 배치에 전체 데이터 분포의 데이터가 다양하게 포함되도록 하는데 도움이 됨 (순서에 의존한 학습x, 하위 그룹의 구조에 민감x)
        tf.util.shuffle(data);

        // Step 2. 텐서로 변환
        // 두개의 배열을 만든다. 하나는 입력 예시(마력 항목) 다른 하나는 실제 출력 값(머신러닝에서 라벨이라고 함)을 위한 배열임
        // 그런 다음 각 배열 데이터를 2D 텐서로 변환함. 텐서의 모양은 [num_examples, num_features_per_example]임
        // 여기에는 inputs.lenth 예시가 있으며 각 예시는 1의 입력 특징(마력)을 가지고 있음
        const inputs = data.map(d => d.horsepower)
        const labels = data.map(d => d.mpg);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        // Step 3. 데이터 정규화
        // 머신러닝을 위한 또 다른 권장사항인 정규화 과정임 (구글에서는 데이터를 정규화 함)
        // 최소-최대 조정을 사용하여 데이터를 숫자 범위 0-1로 정규화 함
        // tensorflow.js로 빌드할 많은 머신러닝 모델 내부는 너무 크지 않은 숫자에 대해 작동하도록 설계되어있기 때문에 정규화가 중요함
        // 일반적인 데이터 정규화 범위는 0 to 1 또는 -1 to 1임
        // 데이터를 합당한 범위로 정규화하는 습관을 들이면 모델의 학습 성공률이 높아짐
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        // 데이터 및 정규화 경계 변환
        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            // Return the min/max bounds so we can use them later.
            inputMax,
            inputMin,
            labelMax,
            labelMin,
            }
    });
}

async function trainModel(model, inputs, labels) {
    // 학습 준비
    model.compile({
        optimizer: tf.train.adam(), // 모델이 예시를 보면서 업데이트하는 데 적용될 알고리즘. 
                                    // 다양한 옵티마이저가 있지만 이 예제에서는 효과적이고 구성이 필요없는 adam 옵티마이저 선택
        loss: tf.losses.meanSquaredError, // 표시되는 각 배치(데이터 하위 집합)를 얼마나 잘 학습하고 있는지 모델에 알려줄 함수. 
                                            // 여기서는 meansquarederror를 사용해 모델이 수행한 예측을 실제 값과 비교함
        metrics: ['mse'],
    });

    const batchSize = 32; // 각 학습 반복에서 모델이 보게 될 데이터 하위 집합의 크기.
                        // 일반적인 배치 크기는 32~512범위임. 모든 문제에 맞는 이상적인 배치 크기는 없으며 다양한 배치 크기의 수학적 연계성이 있음
    const epochs = 50; // 모델이 제공된 전체 데이터 세트를 볼 횟수. 여기서는 데이터 세트를 50번 반복한다.

    // 학습 루프 시작
    return await model.fit(inputs, labels, { // model.fit은 학습 루프를 시작하기 위해 호출하는 비동기 함수. (비동기라 제공된 프라미스를 반환하여 학습이 완료되는 시기를 호출자가 결정할 수 있음)
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}

