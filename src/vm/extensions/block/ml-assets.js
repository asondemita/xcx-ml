/**
 * TensorFlow.js 一式のロード処理。画像分類器と音声分類器で共有する。
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
 */

/**
 * 拡張機能本体の URL から読み込み元が見つからないときの既定値
 * @type {string}
 */
export const DEFAULT_ASSETS_BASE_URL = 'https://asondemita.github.io/xcx-ml/';

const TFJS_PATH = 'libs/tf.min.js';
const KNN_CLASSIFIER_PATH = 'libs/knn-classifier.min.js';

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
 * @param {string} assetsBaseURL - ライブラリの読み込み元 (末尾 / 付き)
 * @returns {Promise} ロード完了で resolve する Promise
 */
export const loadTensorFlow = assetsBaseURL => {
    if (!tensorFlowLoading) {
        tensorFlowLoading = (async () => {
            if (!window.tf) await loadScript(assetsBaseURL + TFJS_PATH);
            if (!window.knnClassifier) await loadScript(assetsBaseURL + KNN_CLASSIFIER_PATH);
        })()
            .catch(error => {
                // 失敗したら次回の呼び出しでリトライできるようにする
                tensorFlowLoading = null;
                throw error;
            });
    }
    return tensorFlowLoading;
};
