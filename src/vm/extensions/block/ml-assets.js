/**
 * TensorFlow.js 一式のロード処理。画像分類器と音声分類器で共有する。
 *
 * ライブラリとモデルは拡張機能と同じ GitHub Pages から配信する。
 * 学校のネットワークフィルタは CDN (cdn.jsdelivr.net) や TensorFlow Hub を
 * ブロックしていることが多く、拡張機能本体が読める配信元に同梱しておけば
 * 追加の許可なしで動くため。ファイルは libs/ と models/ に置いてある。
 */

export const ASSETS_BASE_URL = 'https://asondemita.github.io/xcx-ml';

const TFJS_URL = `${ASSETS_BASE_URL}/libs/tf.min.js`;
const KNN_CLASSIFIER_URL = `${ASSETS_BASE_URL}/libs/knn-classifier.min.js`;

/**
 * <script> タグでスクリプトをロードする。
 * @param {string} url - スクリプトの URL
 * @returns {Promise} ロード完了で resolve する Promise
 */
export const loadScript = url => new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = url;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load script: ${url}`));
    document.head.appendChild(script);
});

/**
 * 実行中の TensorFlow.js ロード処理。複数の分類器から同時に呼ばれても 1 回にまとめる。
 * @type {?Promise}
 */
let tensorFlowLoading = null;

/**
 * TensorFlow.js と KNN 分類器ライブラリをロードする (未ロードのときだけ)。
 * @returns {Promise} ロード完了で resolve する Promise
 */
export const loadTensorFlow = () => {
    if (!tensorFlowLoading) {
        tensorFlowLoading = (async () => {
            if (!window.tf) await loadScript(TFJS_URL);
            if (!window.knnClassifier) await loadScript(KNN_CLASSIFIER_URL);
        })()
            .catch(error => {
                // 失敗したら次回の呼び出しでリトライできるようにする
                tensorFlowLoading = null;
                throw error;
            });
    }
    return tensorFlowLoading;
};
