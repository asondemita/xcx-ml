/**
 * 音声分類器 (speech-commands モデルの特徴量 + KNN による転移学習)。
 *
 * マイク入力を WebAudio の AnalyserNode でスペクトログラム (約 1 秒 = 43 フレーム × 232 ビン)
 * にし、TensorFlow.js speech-commands の学習済みモデル (browser_fft, 18 words) の
 * 隠れ Dense 層 (2000 次元) の出力を特徴量として KNN で分類する。
 * speech-commands ライブラリ自体は使わず、同じ前処理をここで行う
 * (ライブラリ版は転移学習に数秒の学習処理が必要で、クリック即学習の KNN 方式に合わないため)。
 *
 * モデルは https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.3/browser_fft/18w/
 * から取得したものを models/speech_commands_18w/ に同梱している。
 *
 * マイクは最初に使われたときに起動し、停止されるまでスペクトログラムを更新し続ける。
 * フレームの取り込みは ScriptProcessorNode の音声処理コールバックで駆動する。
 * タイマー (setInterval) だとタブがバックグラウンドになったときに毎秒 1 回程度まで
 * 間引かれてフレームが揃わなくなるが、音声処理はタブの状態に関係なく進むため。
 * 分類は呼ばれたときだけ直近 1 秒分で推論する (常時推論ループなし)。
 */

import KnnClassifierBase, {KNN_K} from './knn-classifier-base';
import {ASSETS_BASE_URL, loadTensorFlow} from './ml-assets';

const SPEECH_MODEL_URL = `${ASSETS_BASE_URL}/models/speech_commands_18w/model.json`;

/**
 * 特徴抽出に使うモデルの識別子
 * @type {string}
 */
const MODEL_ID = 'speech_commands_18w';

/**
 * モデルが前提とするサンプリングレート [Hz]
 * @type {number}
 */
const SAMPLE_RATE = 44100;

/**
 * 1 フレームの FFT サイズ (AnalyserNode の fftSize はこの 2 倍)
 * @type {number}
 */
const FFT_SIZE = 1024;

/**
 * 1 回の分類に使うフレーム数 (43 × 1024 / 44100 ≒ 1 秒)
 * @type {number}
 */
const NUM_FRAMES = 43;

/**
 * 1 フレームのうちモデルに渡す周波数ビン数 (低い方から)
 * @type {number}
 */
const FRAME_SIZE = 232;

/**
 * 学習用の録音待ちタイムアウト
 * @type {number} [milliseconds]
 */
const CAPTURE_TIMEOUT = 5000;

/**
 * 正規化時のゼロ除算防止用 (TensorFlow.js の float32 epsilon と同じ)
 * @type {number}
 */
const EPSILON = 1e-7;

export default class AudioClassifier extends KnnClassifierBase {

    constructor () {
        super(MODEL_ID);

        /**
         * 特徴抽出モデル (speech-commands モデルを隠れ Dense 層で切ったもの)
         */
        this.embeddingModel = null;

        /**
         * マイク入力のストリーム
         * @type {?MediaStream}
         */
        this.stream = null;

        /**
         * マイク処理用の AudioContext
         * @type {?AudioContext}
         */
        this.audioContext = null;

        /**
         * スペクトログラム取得用の AnalyserNode
         * @type {?AnalyserNode}
         */
        this.analyser = null;

        /**
         * フレーム取り込みを駆動する ScriptProcessorNode。マイク起動中は非 null。
         * @type {?ScriptProcessorNode}
         */
        this.processor = null;

        /**
         * ScriptProcessorNode が使えない環境でのフレーム取得タイマー
         * @type {?number}
         */
        this.frameTimer = null;

        /**
         * 実行中のマイク起動処理
         * @type {?Promise}
         */
        this.microphoneStarting = null;

        /**
         * AnalyserNode から周波数データを受け取るバッファ
         * @type {?Float32Array}
         */
        this.freqData = null;

        /**
         * 直近のフレーム (古い順、最大 NUM_FRAMES 個)
         * @type {Array<Float32Array>}
         */
        this.frames = [];

        /**
         * マイク起動後に取得したフレームの総数
         * @type {number}
         */
        this.frameCount = 0;

        /**
         * 録音完了を待っている処理
         * @type {Array<{target: number, resolve: Function, reject: Function}>}
         */
        this.captureWaiters = [];
    }

    /**
     * TensorFlow.js / speech-commands モデル / KNN 分類器をロードする。
     * @returns {Promise} 準備完了で resolve する Promise
     */
    async loadModel () {
        await loadTensorFlow();
        const tf = window.tf;
        const baseModel = await tf.loadLayersModel(SPEECH_MODEL_URL);
        // 最後から 2 番目の Dense 層の出力を特徴量にする (speech-commands の転移学習と同じ切り出し方)
        const layers = baseModel.layers;
        let index = layers.length - 2;
        while (index >= 0 && layers[index].getClassName().toLowerCase() !== 'dense') {
            index--;
        }
        if (index < 0) {
            throw new Error('hidden dense layer is not found in the speech model');
        }
        this.embeddingModel = tf.model({
            inputs: baseModel.inputs,
            outputs: layers[index].output
        });
        // 初回推論は遅い (シェーダーコンパイルなど) ので、ここで一度走らせておく
        tf.tidy(() => this.embeddingModel.predict(tf.zeros([1, NUM_FRAMES, FRAME_SIZE, 1])));
        this.knn = window.knnClassifier.create();
    }

    /**
     * マイクが起動しているかどうか。
     * @returns {boolean} 起動中なら true
     */
    isListening () {
        return this.processor !== null || this.frameTimer !== null;
    }

    /**
     * マイクを起動してスペクトログラムの更新を始める (起動済みなら何もしない)。
     * @returns {Promise} 起動完了で resolve する Promise
     */
    startMicrophone () {
        if (this.isListening()) return Promise.resolve();
        if (!this.microphoneStarting) {
            this.microphoneStarting = (async () => {
                const stream = await navigator.mediaDevices.getUserMedia({audio: true, video: false});
                try {
                    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
                    let context;
                    try {
                        context = new AudioContextClass({sampleRate: SAMPLE_RATE});
                    } catch (error) {
                        // sampleRate 指定に対応していないブラウザ
                        context = new AudioContextClass();
                    }
                    if (context.sampleRate !== SAMPLE_RATE) {
                        console.warn(`audio sample rate is ${context.sampleRate} Hz (expected ${SAMPLE_RATE} Hz)`);
                    }
                    if (context.state === 'suspended') {
                        // ユーザー操作 (ブロックのクリック) の後なので再開できるはず
                        context.resume().catch(error => console.warn(error));
                    }
                    const analyser = context.createAnalyser();
                    analyser.fftSize = FFT_SIZE * 2;
                    analyser.smoothingTimeConstant = 0;
                    context.createMediaStreamSource(stream).connect(analyser);
                    this.stream = stream;
                    this.audioContext = context;
                    this.analyser = analyser;
                    this.freqData = new Float32Array(analyser.frequencyBinCount);
                    this.frames = [];
                    this.frameCount = 0;
                    if (typeof context.createScriptProcessor === 'function') {
                        // FFT_SIZE サンプルごと (約 23ms) に呼ばれる。出力は書かないので無音。
                        const processor = context.createScriptProcessor(FFT_SIZE, 1, 1);
                        processor.onaudioprocess = () => this.onAudioFrame();
                        analyser.connect(processor);
                        processor.connect(context.destination);
                        this.processor = processor;
                    } else {
                        this.frameTimer = setInterval(
                            () => this.onAudioFrame(),
                            FFT_SIZE / context.sampleRate * 1000
                        );
                    }
                } catch (error) {
                    stream.getTracks().forEach(track => track.stop());
                    throw error;
                }
            })()
                .finally(() => {
                    this.microphoneStarting = null;
                });
        }
        return this.microphoneStarting;
    }

    /**
     * マイクを停止して、保持しているフレームを捨てる。
     */
    stopMicrophone () {
        if (this.processor) {
            this.processor.onaudioprocess = null;
            this.processor.disconnect();
            this.processor = null;
        }
        if (this.frameTimer !== null) {
            clearInterval(this.frameTimer);
            this.frameTimer = null;
        }
        const waiters = this.captureWaiters;
        this.captureWaiters = [];
        waiters.forEach(waiter => waiter.reject(new Error('microphone is stopped')));
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.audioContext) {
            this.audioContext.close().catch(() => {});
            this.audioContext = null;
        }
        this.analyser = null;
        this.freqData = null;
        this.frames = [];
        this.frameCount = 0;
    }

    /**
     * 1 フレーム分の周波数データを取り込む。
     */
    onAudioFrame () {
        if (!this.analyser) return;
        this.analyser.getFloatFrequencyData(this.freqData);
        if (this.freqData[0] === -Infinity) {
            // 入力がまだ無い (完全な無音) フレームは捨てる
            return;
        }
        this.frames.push(this.freqData.slice(0, FRAME_SIZE));
        if (this.frames.length > NUM_FRAMES) {
            this.frames.shift();
        }
        this.frameCount++;
        if (this.captureWaiters.length > 0) {
            const done = this.captureWaiters.filter(waiter => this.frameCount >= waiter.target);
            if (done.length > 0) {
                this.captureWaiters = this.captureWaiters.filter(waiter => !done.includes(waiter));
                const frames = this.frames.slice();
                done.forEach(waiter => waiter.resolve(frames));
            }
        }
    }

    /**
     * これから約 1 秒分のフレームを録音する。
     * @returns {Promise<Array<Float32Array>>} 録音したフレームで resolve する Promise
     */
    captureNext () {
        return new Promise((resolve, reject) => {
            const waiter = {
                target: this.frameCount + NUM_FRAMES,
                resolve,
                reject
            };
            this.captureWaiters.push(waiter);
            setTimeout(() => {
                if (this.captureWaiters.includes(waiter)) {
                    this.captureWaiters = this.captureWaiters.filter(w => w !== waiter);
                    reject(new Error('timeout to capture audio'));
                }
            }, CAPTURE_TIMEOUT);
        });
    }

    /**
     * フレーム列から特徴量を計算する。
     * @param {Array<Float32Array>} frames - NUM_FRAMES 個のフレーム
     * @returns {Tensor} 特徴量テンソル (呼び出し側で dispose すること)
     */
    embed (frames) {
        const tf = window.tf;
        return tf.tidy(() => {
            const data = new Float32Array(NUM_FRAMES * FRAME_SIZE);
            frames.forEach((frame, i) => data.set(frame, i * FRAME_SIZE));
            const x = tf.tensor4d(data, [1, NUM_FRAMES, FRAME_SIZE, 1]);
            // speech-commands と同じく全体の平均 0・分散 1 に正規化する
            const {mean, variance} = tf.moments(x);
            const normalized = x.sub(mean).div(variance.sqrt().add(EPSILON));
            return this.embeddingModel.predict(normalized);
        });
    }

    /**
     * モデルのロードとマイクの起動を済ませる (学習前の準備)。
     * @returns {Promise} 準備完了で resolve する Promise
     */
    async prepare () {
        await this.load();
        await this.startMicrophone();
    }

    /**
     * これから約 1 秒間の音を指定ラベルの例として学習する。
     * @param {string} label - ラベル
     * @returns {Promise} 学習完了で resolve する Promise
     */
    async train (label) {
        await this.prepare();
        const frames = await this.captureNext();
        const features = this.embed(frames);
        // addExample は内部でコピーを保持するので、渡したテンソルはすぐ破棄してよい
        this.knn.addExample(features, label);
        features.dispose();
    }

    /**
     * 直近約 1 秒間の音を分類してラベルを返す。
     * マイク起動直後でまだ 1 秒分のフレームが無いときは空文字列を返す。
     * @returns {Promise<string>} 分類されたラベル (未学習なら空文字列)
     */
    async classify () {
        if (!this.hasExamples()) return '';
        await this.startMicrophone();
        if (this.frames.length < NUM_FRAMES) return '';
        const features = this.embed(this.frames);
        try {
            const result = await this.knn.predictClass(features, KNN_K);
            return result.label;
        } finally {
            features.dispose();
        }
    }
}
