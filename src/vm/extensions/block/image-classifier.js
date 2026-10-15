/**
 * 画像分類器 (MobileNet 特徴量 + KNN による転移学習)。
 *
 * TensorFlow.js 一式は最初に使われたときに動的にロードするので、
 * ML ブロックを使わないプロジェクトではメモリもネットワークも消費しない。
 *
 * 推論のたびに生成される特徴量テンソルを必ず dispose してメモリリークを防ぐ。
 * 分類の常時ループも持たない (呼ばれたときだけ推論する)。
 */

import KnnClassifierBase, {KNN_K} from './knn-classifier-base';
import {DEFAULT_ASSETS_BASE_URL, loadScript, loadTensorFlow} from './ml-assets';

const MOBILENET_PATH = 'libs/mobilenet.min.js';
const MOBILENET_MODEL_PATH = 'models/mobilenet_v2_050_224/model.json';

/**
 * 特徴抽出に使うモデルの識別子
 * @type {string}
 */
const MODEL_ID = 'mobilenet_v2_050_224';

export default class ImageClassifier extends KnnClassifierBase {

    /**
     * @param {string} [assetsBaseURL] - ライブラリ・モデルの読み込み元 (末尾 / 付き)
     */
    constructor (assetsBaseURL = DEFAULT_ASSETS_BASE_URL) {
        super(MODEL_ID, assetsBaseURL);

        /**
         * MobileNet モデル (特徴抽出器)
         */
        this.mobileNet = null;
    }

    /**
     * TensorFlow.js / MobileNet / KNN 分類器をロードする。
     * @returns {Promise} 準備完了で resolve する Promise
     */
    async loadModel () {
        const base = this.assetsBaseURL;
        await loadTensorFlow(base);
        if (!window.mobilenet) await loadScript(base + MOBILENET_PATH);
        // alpha 0.5: 既定 (1.0) よりモデルが小さく、KNN 用途では精度も十分
        // 重みファイルは TensorFlow.js が model.json と同じ場所から読む
        this.mobileNet = await window.mobilenet.load({
            version: 2,
            alpha: 0.5,
            modelUrl: base + MOBILENET_MODEL_PATH
        });
        this.knn = window.knnClassifier.create();
    }

    /**
     * 現在の画像を指定ラベルの例として学習する。
     * @param {HTMLVideoElement|HTMLCanvasElement} input - 入力画像
     * @param {string} label - ラベル
     * @returns {Promise} 学習完了で resolve する Promise
     */
    async train (input, label) {
        await this.load();
        const features = window.tf.tidy(() => this.mobileNet.infer(input, true));
        // addExample は内部でコピーを保持するので、渡したテンソルはすぐ破棄してよい
        this.knn.addExample(features, label);
        features.dispose();
    }

    /**
     * 現在の画像を分類してラベルを返す。
     * @param {HTMLVideoElement|HTMLCanvasElement} input - 入力画像
     * @returns {Promise<string>} 分類されたラベル (未学習なら空文字列)
     */
    async classify (input) {
        await this.load();
        if (!this.hasExamples()) return '';
        const features = window.tf.tidy(() => this.mobileNet.infer(input, true));
        try {
            const result = await this.knn.predictClass(features, KNN_K);
            return result.label;
        } finally {
            features.dispose();
        }
    }
}
