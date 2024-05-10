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

    
    const model = createModel();
    tfvis.show.modelSummary({name: 'Model Summary'}, model);
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

