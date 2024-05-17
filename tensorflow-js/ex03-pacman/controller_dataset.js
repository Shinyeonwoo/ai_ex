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

/**
 * 사용자가 예제 Tensor를 추가할 수 있는 웹캠 컨트롤용 데이터세트
 * 특정 라벨의 경우. 이 객체는 두 개의 큰 x와 y를 연결합니다.
 */
export class ControllerDataset {
  constructor(numClasses) {
    this.numClasses = numClasses;
  }

/**
   * 컨트롤러 데이터 세트에 예제를 추가합니다.
   * @param {Tensor} 예제 예제를 나타내는 텐서입니다. 이미지일 수도 있고,
   * 활성화 또는 다른 유형의 Tensor.
   * @param {number} label 예제의 레이블입니다. 숫자여야 합니다.
   */
  addExample(example, label) {
    // 레이블을 원-핫 인코딩합니다.
    const y = tf.tidy(
        () => tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses));

    if (this.xs == null) {
      // 추가되는 첫 번째 예제의 경우 example과 y를 유지하여
      // ControllerDataset는 입력 메모리를 소유합니다. 이는 다음을 보장합니다.
      // tf.tidy()에서 addExample()이 호출되면 이 Tensor는 가져오지 않습니다.
      // 폐기됨.
      this.xs = tf.keep(example);
      this.ys = tf.keep(y);
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));

      const oldY = this.ys;
      this.ys = tf.keep(oldY.concat(y, 0));

      oldX.dispose();
      oldY.dispose();
      y.dispose();
    }
  }
}