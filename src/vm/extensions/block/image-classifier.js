/**
 * 画像分類器 (MobileNet 特徴量 + KNN による転移学習)。
 *
 * TensorFlow.js 一式は最初に使われたときに動的にロードするので、
 * ML ブロックを使わないプロジェクトではメモリもネットワークも消費しない。
 *
 * ライブラリとモデルは拡張機能と同じ GitHub Pages から配信する。
 * 学校のネットワークフィルタは CDN (cdn.jsdelivr.net) や TensorFlow Hub を
 * ブロックしていることが多く、拡張機能本体が読める配信元に同梱しておけば
 * 追加の許可なしで動くため。ファイルは libs/ と models/ に置いてある。
 *
 * 読み込み元は固定せず、拡張機能本体が実際に読み込まれた URL を基準にする。
 * Xcratch の配布サイトによっては拡張機能をプロキシ (例:
 * https://xcratch.699.jp/ext/?url=asondemita/xcx-ml/dist/xcx-ml.mjs) 経由で
 * 配信しており、その環境では GitHub Pages 自体がフィルタで遮られていることが
 * あるため、ライブラリとモデルも同じプロキシを通して読む。
 *
 * ML2Scratch と違い、推論のたびに生成される特徴量テンソルを必ず dispose して
 * メモリリークを防ぐ。分類の常時ループも持たない（呼ばれたときだけ推論する）。
 */

/**
 * 拡張機能本体の URL から読み込み元 (ライブラリ・モデル) が見つからないときの既定値
 * @type {string}
 */
const DEFAULT_ASSETS_BASE_URL = 'https://asondemita.github.io/xcx-ml/';

const TFJS_PATH = 'libs/tf.min.js';
const KNN_CLASSIFIER_PATH = 'libs/knn-classifier.min.js';
const MOBILENET_PATH = 'libs/mobilenet.min.js';
const MOBILENET_MODEL_PATH = 'models/mobilenet_v2_050_224/model.json';

/**
 * 拡張機能本体の URL から、ライブラリ・モデルの読み込み元 (末尾 / 付き) を求める。
 * 本体は <base>/dist/xcx-ml.mjs に置かれているので、末尾の dist/… を取り除く。
 * プロキシ経由 (…/ext/?url=asondemita/xcx-ml/dist/xcx-ml.mjs) でも
 * 同じ規則で …/ext/?url=asondemita/xcx-ml/ になり、ライブラリとモデルも
 * プロキシを通る。形が合わなければ既定の GitHub Pages を使う。
 * @param {string} extensionURL - 拡張機能本体が読み込まれた URL
 * @returns {string} 読み込み元の URL
 */
export const assetsBaseURLFrom = extensionURL => {
    const matched = /^(.*\/)dist\/[^/]+\.mjs$/.exec(extensionURL || '');
    return matched ? matched[1] : DEFAULT_ASSETS_BASE_URL;
};

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

    /**
     * @param {string} [assetsBaseURL] - ライブラリ・モデルの読み込み元 (末尾 / 付き)
     */
    constructor (assetsBaseURL = DEFAULT_ASSETS_BASE_URL) {
        /**
         * ライブラリ・モデルの読み込み元
         * @type {string}
         */
        this.assetsBaseURL = assetsBaseURL;

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
                const base = this.assetsBaseURL;
                if (!window.tf) await loadScript(base + TFJS_PATH);
                if (!window.knnClassifier) await loadScript(base + KNN_CLASSIFIER_PATH);
                if (!window.mobilenet) await loadScript(base + MOBILENET_PATH);
                // alpha 0.5: 既定 (1.0) よりモデルが小さく、KNN 用途では精度も十分
                // 重みファイルは TensorFlow.js が model.json と同じ場所から読む
                this.mobileNet = await window.mobilenet.load({
                    version: 2,
                    alpha: 0.5,
                    modelUrl: base + MOBILENET_MODEL_PATH
                });
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
     * ライブラリとモデルのロードが完了しているかどうか。
     * @returns {boolean} 準備完了なら true
     */
    isReady () {
        return !!this.mobileNet && !!this.knn;
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
