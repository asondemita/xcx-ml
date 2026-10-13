/**
 * 特徴抽出モデル + KNN 分類器の共通部分。
 *
 * サブクラスは loadModel() で特徴抽出モデルと KNN 分類器を用意し、
 * 入力に応じた train() / classify() を実装する。
 * 学習データの消去・書き出し・復元はここで共通に扱う。
 */

/**
 * KNN 分類時に参照する近傍数
 * @type {number}
 */
export const KNN_K = 3;

export default class KnnClassifierBase {

    /**
     * @param {string} modelId - 特徴抽出に使うモデルの識別子。
     * モデルが変わると特徴量の互換性がなくなるため、学習データに記録して復元時に検査する。
     */
    constructor (modelId) {
        /**
         * 特徴抽出モデルの識別子
         * @type {string}
         */
        this.modelId = modelId;

        /**
         * ライブラリとモデルのロード処理。一度だけ実行する。
         * @type {?Promise}
         */
        this.loading = null;

        /**
         * ロードが完了しているかどうか
         * @type {boolean}
         */
        this.ready = false;

        /**
         * KNN 分類器
         */
        this.knn = null;
    }

    /**
     * ライブラリとモデルをロードする (未ロードのときだけ)。
     * @returns {Promise} 準備完了で resolve する Promise
     */
    load () {
        if (!this.loading) {
            this.loading = this.loadModel()
                .then(() => {
                    this.ready = true;
                })
                .catch(error => {
                    // 失敗したら次回の呼び出しでリトライできるようにする
                    this.loading = null;
                    throw error;
                });
        }
        return this.loading;
    }

    /**
     * 特徴抽出モデルと KNN 分類器を用意する。サブクラスで実装する。
     * @returns {Promise} 準備完了で resolve する Promise
     */
    loadModel () {
        return Promise.reject(new Error('loadModel() is not implemented'));
    }

    /**
     * ライブラリとモデルのロードが完了しているかどうか。
     * @returns {boolean} 準備完了なら true
     */
    isReady () {
        return this.ready;
    }

    /**
     * 学習済みの例があるかどうか。
     * @returns {boolean} 1 つ以上のラベルが学習済みなら true
     */
    hasExamples () {
        return !!this.knn && this.knn.getNumClasses() > 0;
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
     * 学習データを JSON 化できる形で書き出す。
     * @returns {{model: string, labels: object}} 学習データ
     */
    exportDataset () {
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
        return {
            model: this.modelId,
            labels: labels
        };
    }

    /**
     * 学習データがこの分類器で復元できる形かどうか検査する。
     * @param {object} data - exportDataset() 形式の学習データ
     * @throws {Error} 復元できない形のとき
     */
    checkDataset (data) {
        if (!data || typeof data.labels !== 'object' || data.labels === null) {
            throw new Error('invalid learning data format');
        }
        const labels = Object.keys(data.labels);
        if (labels.length === 0) return;
        if (data.model !== this.modelId) {
            // 別のモデルで作った特徴量は互換性がない
            throw new Error(`incompatible model: ${data.model}`);
        }
        for (const label of labels) {
            const examples = data.labels[label];
            if (!examples || !Array.isArray(examples.shape) || !Array.isArray(examples.data)) {
                throw new Error(`invalid examples for label: ${label}`);
            }
        }
    }

    /**
     * 書き出した学習データを復元する。今の学習内容は破棄される。
     * 例が 1 つも無いデータのときはモデルをロードせず、学習内容の消去だけ行う。
     * @param {object} data - exportDataset() 形式の学習データ
     * @returns {Promise} 復元完了で resolve する Promise
     */
    async importDataset (data) {
        this.checkDataset(data);
        if (Object.keys(data.labels).length === 0) {
            this.reset();
            return;
        }
        await this.load();
        const dataset = {};
        for (const label of Object.keys(data.labels)) {
            const examples = data.labels[label];
            dataset[label] = window.tf.tensor2d(examples.data, examples.shape);
        }
        this.knn.clearAllClasses();
        this.knn.setClassifierDataset(dataset);
    }
}
