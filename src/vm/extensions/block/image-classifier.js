/**
 * 画像分類器 (MobileNet 特徴量 + KNN による転移学習)。
 *
 * TensorFlow.js 一式は最初に使われたときに CDN から動的にロードするので、
 * ML ブロックを使わないプロジェクトではメモリもネットワークも消費しない。
 *
 * ML2Scratch と違い、推論のたびに生成される特徴量テンソルを必ず dispose して
 * メモリリークを防ぐ。分類の常時ループも持たない（呼ばれたときだけ推論する）。
 */

const TFJS_URL = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
const KNN_CLASSIFIER_URL = 'https://cdn.jsdelivr.net/npm/@tensorflow-models/knn-classifier@1.2.6/dist/knn-classifier.min.js';
const MOBILENET_URL = 'https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.1/dist/mobilenet.min.js';

/**
 * KNN 分類時に参照する近傍数
 * @type {number}
 */
const KNN_K = 3;

/**
 * 学習データファイルのフォーマット識別子
 * @type {string}
 */
const DATA_FORMAT = 'xcx-g2s-knn';

/**
 * 特徴抽出に使うモデルの識別子。
 * モデルが変わると特徴量の互換性がなくなるため、学習データファイルに記録して
 * 読み込み時に検査する。
 * @type {string}
 */
const MODEL_ID = 'mobilenet_v2_050_224';

/**
 * <script> タグでスクリプトをロードする。
 * @param {string} url - スクリプトの URL
 * @returns {Promise} ロード完了で resolve する Promise
 */
const loadScript = url => new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = url;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load script: ${url}`));
    document.head.appendChild(script);
});

export default class ImageClassifier {

    constructor () {
        /**
         * ライブラリとモデルのロード処理。一度だけ実行する。
         * @type {?Promise}
         */
        this.loading = null;

        /**
         * MobileNet モデル (特徴抽出器)
         */
        this.mobileNet = null;

        /**
         * KNN 分類器
         */
        this.knn = null;
    }

    /**
     * TensorFlow.js / MobileNet / KNN 分類器をロードする。
     * @returns {Promise} 準備完了で resolve する Promise
     */
    load () {
        if (!this.loading) {
            this.loading = (async () => {
                if (!window.tf) await loadScript(TFJS_URL);
                if (!window.knnClassifier) await loadScript(KNN_CLASSIFIER_URL);
                if (!window.mobilenet) await loadScript(MOBILENET_URL);
                // alpha 0.5: 既定 (1.0) よりモデルが小さく、KNN 用途では精度も十分
                this.mobileNet = await window.mobilenet.load({version: 2, alpha: 0.5});
                this.knn = window.knnClassifier.create();
            })()
                .catch(error => {
                    // 失敗したら次回の呼び出しでリトライできるようにする
                    this.loading = null;
                    throw error;
                });
        }
        return this.loading;
    }

    /**
     * 学習済みの例があるかどうか。
     * @returns {boolean} 1 つ以上のラベルが学習済みなら true
     */
    hasExamples () {
        return !!this.knn && this.knn.getNumClasses() > 0;
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

    /**
     * 学習した例をすべて消去する。保持していたテンソルも解放される。
     */
    reset () {
        if (this.knn) {
            this.knn.clearAllClasses();
        }
    }

    /**
     * 学習データを JSON 文字列にシリアライズする。
     * @returns {string} 学習データの JSON 文字列
     */
    serialize () {
        const labels = {};
        if (this.knn) {
            const dataset = this.knn.getClassifierDataset();
            for (const label of Object.keys(dataset)) {
                const examples = dataset[label];
                labels[label] = {
                    shape: examples.shape,
                    data: Array.from(examples.dataSync())
                };
            }
        }
        return JSON.stringify({
            format: DATA_FORMAT,
            version: 1,
            model: MODEL_ID,
            labels: labels
        });
    }

    /**
     * シリアライズされた学習データを復元する。今の学習内容は破棄される。
     * @param {object} parsed - JSON.parse 済みの学習データ
     * @returns {Promise} 復元完了で resolve する Promise
     */
    async restore (parsed) {
        if (!parsed || parsed.format !== DATA_FORMAT || !parsed.labels) {
            throw new Error('invalid learning data format');
        }
        if (parsed.model !== MODEL_ID) {
            // 別のモデルで作った特徴量は互換性がない
            throw new Error(`incompatible model: ${parsed.model}`);
        }
        await this.load();
        const dataset = {};
        for (const label of Object.keys(parsed.labels)) {
            const examples = parsed.labels[label];
            dataset[label] = window.tf.tensor2d(examples.data, examples.shape);
        }
        this.knn.clearAllClasses();
        this.knn.setClassifierDataset(dataset);
    }
}
