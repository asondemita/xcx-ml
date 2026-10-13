import BlockType from '../../extension-support/block-type';
import ArgumentType from '../../extension-support/argument-type';
import translations from './translations.json';
import blockIcon from './block-icon.png';

import ImageClassifier from './image-classifier';
import AudioClassifier from './audio-classifier';

/**
 * Formatter which is used for translation.
 * This will be replaced which is used in the runtime.
 * @param {object} messageData - format-message object
 * @returns {string} - message for the locale
 */
let formatMessage = messageData => messageData.defaultMessage;

/**
 * Setup format-message for this extension.
 */
const setupTranslations = () => {
    const localeSetup = formatMessage.setup();
    if (localeSetup && localeSetup.translations[localeSetup.locale]) {
        Object.assign(
            localeSetup.translations[localeSetup.locale],
            translations[localeSetup.locale]
        );
    }
};

const EXTENSION_ID = 'xcxml';

/**
 * URL to get this extension as a module.
 * When it was loaded as a module, 'extensionURL' will be replaced a URL which is retrieved from.
 * @type {string}
 */
let extensionURL = 'https://asondemita.github.io/xcx-ml/dist/xcx-ml.mjs';

/**
 * 学習データファイルのフォーマット識別子
 * @type {string}
 */
const DATA_FORMAT = 'xcx-g2s-knn';

/**
 * 学習データファイルのバージョン。
 * 1: 画像の学習データのみ (model, labels)
 * 2: 1 に音声の学習データ (audio: {model, labels}) を追加。1 の読み手でも画像部分は読める。
 * @type {number}
 */
const DATA_VERSION = 2;

/**
 * 初回セットアップ通知ダイアログの最低表示時間。
 * @type {number} [milliseconds]
 */
const SETUP_NOTICE_MIN_DURATION = 3000;

/**
 * 失敗メッセージの表示時間。
 * @type {number} [milliseconds]
 */
const NOTICE_ERROR_DURATION = 5000;

/**
 * 学習・分類の入力元。
 * @readonly
 * @enum {string}
 */
const InputMode = {
    /** カメラ画像 */
    CAMERA: 'camera',

    /** マイクの音 */
    MICROPHONE: 'microphone'
};

/**
 * States the video sensing activity can be set to.
 * @readonly
 * @enum {string}
 */
const VideoState = {
    /** Video turned off. */
    OFF: 'off',

    /** Video turned on with default y axis mirroring. */
    ON: 'on',

    /** Video turned on without default y axis mirroring. */
    ON_FLIPPED: 'on-flipped'
};

/**
 * Scratch 3.0 blocks for machine learning (KNN image / sound classification).
 */
class ExtensionBlocks {

    /**
     * A translation object which is used in this class.
     * @param {FormatObject} formatter - translation object
     */
    static set formatMessage (formatter) {
        formatMessage = formatter;
        if (formatMessage) setupTranslations();
    }

    /**
     * @return {string} - the name of this extension.
     */
    static get EXTENSION_NAME () {
        return formatMessage({
            id: 'xcxml.name',
            default: 'Machine Learning',
            description: 'name of the extension'
        });
    }

    /**
     * @return {string} - the ID of this extension.
     */
    static get EXTENSION_ID () {
        return EXTENSION_ID;
    }

    /**
     * URL to get this extension.
     * @type {string}
     */
    static get extensionURL () {
        return extensionURL;
    }

    /**
     * Set URL to get this extension.
     * The extensionURL will be changed to the URL of the loading server.
     * @param {string} url - URL
     */
    static set extensionURL (url) {
        extensionURL = url;
    }

    /**
     * Construct a set of blocks for machine learning.
     * @param {Runtime} runtime - the Scratch 3.0 runtime.
     */
    constructor (runtime) {
        /**
         * The Scratch 3.0 runtime.
         * @type {Runtime}
         */
        this.runtime = runtime;

        if (runtime.formatMessage) {
            // Replace 'formatMessage' to a formatter which is used in the runtime.
            formatMessage = runtime.formatMessage;
        }

        /**
         * 画像分類器 (MobileNet + KNN)。ML ブロックが最初に使われたときに生成する。
         * @type {?ImageClassifier}
         */
        this.imageClassifier = null;

        /**
         * 音声分類器 (speech-commands + KNN)。マイクモードで最初に使われたときに生成する。
         * @type {?AudioClassifier}
         */
        this.audioClassifier = null;

        /**
         * 学習・分類の入力元。既存プロジェクトとの互換性のため初期値はカメラ。
         * @type {InputMode}
         */
        this.inputMode = InputMode.CAMERA;

        /**
         * 実行中の分類処理。モニターなどからの連続呼び出しを 1 つにまとめる。
         * @type {?Promise<string>}
         */
        this.mlClassifyRequest = null;

        /**
         * 最後に分類されたラベル。
         * @type {string}
         */
        this.mlLastLabel = '';

        /**
         * 最後に分類した時刻。
         * @type {number} [milliseconds]
         */
        this.mlLabelUpdatedTime = 0;

        /**
         * 分類結果のキャッシュ有効時間。
         * @type {number} [milliseconds]
         */
        this.mlLabelUpdateIntervalTime = 200;

        /**
         * 学習データ読み込みダイアログが開いているかどうか。
         * @type {boolean}
         */
        this.mlLoadDialogOpened = false;

        /**
         * エラーダイアログが開いているかどうか。
         * @type {boolean}
         */
        this.errorDialogOpened = false;

        /**
         * 表示中の通知ダイアログ (セットアップ中・録音中など)。
         * @type {?{dialog: HTMLDialogElement, label: Text, openedAt: number, minDuration: number}}
         */
        this.notice = null;
    }

    /**
     * 画像分類器を返す。まだ無ければ生成する。
     * @returns {ImageClassifier} 画像分類器
     */
    getImageClassifier () {
        if (!this.imageClassifier) {
            this.imageClassifier = new ImageClassifier();
        }
        return this.imageClassifier;
    }

    /**
     * 音声分類器を返す。まだ無ければ生成する。
     * @returns {AudioClassifier} 音声分類器
     */
    getAudioClassifier () {
        if (!this.audioClassifier) {
            this.audioClassifier = new AudioClassifier();
        }
        return this.audioClassifier;
    }

    /**
     * 今の入力モードに対応する分類器を返す (未生成なら null)。
     * @returns {?ImageClassifier|AudioClassifier} 分類器
     */
    getCurrentClassifier () {
        return this.inputMode === InputMode.MICROPHONE ? this.audioClassifier : this.imageClassifier;
    }

    /**
     * 分類結果のキャッシュを捨てる。
     */
    clearLabelCache () {
        this.mlLastLabel = '';
        this.mlLabelUpdatedTime = 0;
    }

    /**
     * ビデオ入力を有効にして、フレームが取得できる状態の video 要素を返す。
     * @returns {Promise<HTMLVideoElement>} ビデオ要素で resolve する Promise
     */
    async getVideoInput () {
        const video = this.runtime.ioDevices.video;
        await video.enableVideo();
        const input = video.provider ? video.provider.video : null;
        if (!input) {
            throw new Error('video input is not available');
        }
        // カメラ起動直後はフレームがまだ無いことがあるので待つ (最大 10 秒)
        for (let i = 0; input.readyState < 2; i++) {
            if (i >= 100) throw new Error('timeout to wait video input ready');
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        return input;
    }

    /**
     * 処理中であることを知らせる非モーダルのダイアログを表示する。
     * 閉じるのは closeNotice()。すでに表示中なら文言だけ差し替える。
     * @param {string} message - 表示する文言
     * @param {number} minDuration - 最低表示時間 [milliseconds]
     */
    openNotice (message, minDuration) {
        if (this.notice) {
            this.updateNotice(message, minDuration);
            return;
        }
        const dialog = document.createElement('dialog');
        dialog.style.padding = '16px';
        // Scratch GUI の UI より手前に、操作を妨げない非モーダルで出す
        dialog.style.position = 'fixed';
        dialog.style.top = '40%';
        dialog.style.zIndex = '1000';
        const label = document.createTextNode(message);
        dialog.appendChild(label);
        document.body.appendChild(dialog);
        dialog.show();
        this.notice = {dialog, label, openedAt: Date.now(), minDuration};
    }

    /**
     * 表示中の通知ダイアログの文言を差し替え、最低表示時間を数え直す。
     * @param {string} message - 表示する文言
     * @param {number} minDuration - 最低表示時間 [milliseconds]
     */
    updateNotice (message, minDuration) {
        const notice = this.notice;
        if (!notice) return;
        notice.label.textContent = message;
        notice.openedAt = Date.now();
        notice.minDuration = minDuration;
    }

    /**
     * 通知ダイアログを閉じる。
     * 成功時は最低表示時間の残りを待ってから閉じる。
     * 失敗時はメッセージを差し替えてしばらく見せてから閉じる。
     * @param {?string} errorMessage - 失敗したときに表示する文言
     */
    closeNotice (errorMessage) {
        const notice = this.notice;
        if (!notice) return;
        this.notice = null;
        let delay = notice.minDuration - (Date.now() - notice.openedAt);
        if (errorMessage) {
            notice.label.textContent = errorMessage;
            delay = NOTICE_ERROR_DURATION;
        }
        setTimeout(() => {
            notice.dialog.close();
            document.body.removeChild(notice.dialog);
        }, Math.max(0, delay));
    }

    /**
     * @returns {string} 初回セットアップ中の文言
     */
    get setupNoticeMessage () {
        return formatMessage({
            id: 'xcxml.setupNotice.message',
            default: 'Preparing the machine learning environment. This may take a while...',
            description: 'notice that the first setup takes a while'
        });
    }

    /**
     * 今の入力モードを指定ラベルの例として学習する。
     * @param {string} label - ラベル
     * @returns {Promise} 学習完了で resolve する Promise
     */
    mlTrain (label) {
        if (this.inputMode === InputMode.MICROPHONE) {
            return this.mlTrainAudio(label);
        }
        return this.mlTrainImage(label);
    }

    /**
     * これから約 1 秒間のマイクの音を指定ラベルの例として学習する。
     * 録音中はダイアログで知らせる。初回はライブラリ・モデルのロードとマイク起動が先に走る。
     * @param {string} label - ラベル
     * @returns {Promise} 学習完了で resolve する Promise
     */
    async mlTrainAudio (label) {
        const classifier = this.getAudioClassifier();
        const firstSetup = !classifier.isReady();
        if (firstSetup) {
            this.openNotice(this.setupNoticeMessage, SETUP_NOTICE_MIN_DURATION);
        }
        try {
            await classifier.prepare();
            this.openNotice(formatMessage({
                id: 'xcxml.listeningNotice.message',
                default: 'Listening to the microphone...',
                description: 'notice that the sound for training is being recorded'
            }), 0);
            await classifier.train(label);
            this.closeNotice();
        } catch (error) {
            this.closeNotice(formatMessage({
                id: 'xcxml.setupNotice.failedMicrophone',
                default: 'Setup failed. Check the network connection and the microphone, then click the block again.',
                description: 'message when the first setup with microphone failed'
            }));
            console.error(error);
        }
    }

    /**
     * 現在のカメラ画像を指定ラベルの例として学習する。
     * 初回はライブラリ・モデルのロードとカメラ起動が走るため、通知ダイアログを表示する。
     * @param {string} label - ラベル
     * @returns {Promise} 学習完了で resolve する Promise
     */
    async mlTrainImage (label) {
        const classifier = this.getImageClassifier();
        const firstSetup = !classifier.isReady();
        if (firstSetup) {
            this.openNotice(this.setupNoticeMessage, SETUP_NOTICE_MIN_DURATION);
        }
        try {
            const input = await this.getVideoInput();
            await classifier.train(input, label);
            if (firstSetup) {
                this.closeNotice();
            }
        } catch (error) {
            if (firstSetup) {
                this.closeNotice(formatMessage({
                    id: 'xcxml.setupNotice.failed',
                    default: 'Setup failed. Check the network connection and the camera, then click the block again.',
                    description: 'message when the first setup failed'
                }));
            }
            console.error(error);
        }
    }

    /**
     * ラベル1を学習する。
     * @returns {Promise} 学習完了で resolve する Promise
     */
    mlTrainLabel1 () {
        return this.mlTrain('1');
    }

    /**
     * ラベル2を学習する。
     * @returns {Promise} 学習完了で resolve する Promise
     */
    mlTrainLabel2 () {
        return this.mlTrain('2');
    }

    /**
     * ラベル3を学習する。
     * @returns {Promise} 学習完了で resolve する Promise
     */
    mlTrainLabel3 () {
        return this.mlTrain('3');
    }

    /**
     * 今の入力 (カメラ画像またはマイクの音) を分類してラベルを返す。
     * 未学習のときはカメラ・マイクを起動せずに空文字列を返す。
     * 短時間に連続で呼ばれた場合 (ステージのモニターなど) はキャッシュを返し、
     * 推論が過剰に走らないようにする。
     * @returns {string|Promise<string>} 分類されたラベル
     */
    mlLabel () {
        const classifier = this.getCurrentClassifier();
        if (!classifier || !classifier.hasExamples()) return '';
        if ((Date.now() - this.mlLabelUpdatedTime) < this.mlLabelUpdateIntervalTime) {
            return this.mlLastLabel;
        }
        if (!this.mlClassifyRequest) {
            const classifying = this.inputMode === InputMode.MICROPHONE ?
                classifier.classify() :
                this.getVideoInput().then(input => classifier.classify(input));
            this.mlClassifyRequest = classifying
                .then(label => {
                    this.mlLastLabel = label;
                    this.mlLabelUpdatedTime = Date.now();
                    return label;
                })
                .catch(error => {
                    console.error(error);
                    return this.mlLastLabel;
                })
                .finally(() => {
                    this.mlClassifyRequest = null;
                });
        }
        return this.mlClassifyRequest;
    }

    /**
     * 学習した内容 (画像・音声の両方) をすべてリセットする。
     */
    mlReset () {
        if (this.imageClassifier) {
            this.imageClassifier.reset();
        }
        if (this.audioClassifier) {
            this.audioClassifier.reset();
        }
        this.clearLabelCache();
    }

    /**
     * 学習データ (画像・音声の両方) を JSON ファイルとして保存 (ダウンロード) する。
     */
    mlSaveData () {
        const imageTrained = this.imageClassifier && this.imageClassifier.hasExamples();
        const audioTrained = this.audioClassifier && this.audioClassifier.hasExamples();
        if (!imageTrained && !audioTrained) {
            // 学習データが無いときは何もしない
            return;
        }
        // 画像のデータはバージョン 1 と同じ位置 (model, labels) に置き、古い読み手との互換性を保つ
        const json = JSON.stringify(Object.assign(
            {
                format: DATA_FORMAT,
                version: DATA_VERSION
            },
            this.getImageClassifier().exportDataset(),
            {
                audio: this.getAudioClassifier().exportDataset()
            }
        ));
        const blob = new Blob([json], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = `xcx-ml-${Date.now()}.json`;
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
        URL.revokeObjectURL(url);
    }

    /**
     * ファイル選択ダイアログを開いて学習データを読み込む。
     * 今の学習内容は読み込んだ内容で置き換えられる。
     * @returns {Promise} 読み込み完了またはキャンセルで resolve する Promise
     */
    async mlLoadData () {
        if (this.mlLoadDialogOpened) {
            // prevent to open multiple dialogs
            return;
        }
        this.mlLoadDialogOpened = true;
        const inputDialog = document.createElement('dialog');
        inputDialog.style.padding = '0px';
        const dialogFace = document.createElement('div');
        dialogFace.style.padding = '16px';
        inputDialog.appendChild(dialogFace);
        const label = document.createTextNode(formatMessage({
            id: 'xcxml.loadDialog.message',
            default: 'select a learning data file',
            description: 'label of learning data loading dialog'
        }));
        dialogFace.appendChild(label);
        // File input
        const fileForm = document.createElement('form');
        fileForm.setAttribute('method', 'dialog');
        fileForm.style.margin = '8px';
        fileForm.addEventListener('submit', e => {
            e.preventDefault();
        });
        dialogFace.appendChild(fileForm);
        const fileInput = document.createElement('input');
        fileInput.setAttribute('type', 'file');
        fileInput.setAttribute('accept', 'application/json,.json');
        fileForm.appendChild(fileInput);
        // Cancel button
        const cancelButton = document.createElement('button');
        cancelButton.textContent = formatMessage({
            id: 'xcxml.loadDialog.cancel',
            default: 'cancel',
            description: 'cancel button on learning data loading dialog'
        });
        cancelButton.style.margin = '8px';
        dialogFace.appendChild(cancelButton);
        // Load button
        const confirmButton = document.createElement('button');
        confirmButton.textContent = formatMessage({
            id: 'xcxml.loadDialog.load',
            default: 'load',
            description: 'load button on learning data loading dialog'
        });
        confirmButton.style.margin = '8px';
        dialogFace.appendChild(confirmButton);
        const file = await new Promise(resolve => {
            confirmButton.onclick = () => {
                if (fileInput.files.length === 0) return;
                resolve(fileInput.files[0]);
            };
            cancelButton.onclick = () => {
                resolve(null);
            };
            inputDialog.addEventListener('keydown', e => {
                if (e.code === 'Escape') {
                    resolve(null);
                }
            });
            document.body.appendChild(inputDialog);
            inputDialog.showModal();
        })
            .finally(() => {
                document.body.removeChild(inputDialog);
                this.mlLoadDialogOpened = false;
            });
        if (!file) return;
        try {
            await this.restoreLearningData(JSON.parse(await file.text()));
            this.clearLabelCache();
        } catch (error) {
            console.error(error);
            this.openErrorDialog(formatMessage({
                id: 'xcxml.loadInvalidFile',
                default: 'This file is not learning data for this extension.',
                description: 'error message for invalid learning data file'
            }));
        }
    }

    /**
     * 学習データファイルの内容を画像・音声の分類器に復元する。
     * ファイルに含まれない側の学習内容は消去する。
     * @param {object} parsed - JSON.parse 済みの学習データ
     * @returns {Promise} 復元完了で resolve する Promise
     */
    async restoreLearningData (parsed) {
        if (!parsed || parsed.format !== DATA_FORMAT || (!parsed.labels && !parsed.audio)) {
            throw new Error('invalid learning data format');
        }
        const imageData = {model: parsed.model, labels: parsed.labels || {}};
        const audioData = parsed.audio || {labels: {}};
        const imageClassifier = this.getImageClassifier();
        const audioClassifier = this.getAudioClassifier();
        // 片方だけ復元されて中途半端にならないよう、先に両方を検査する
        imageClassifier.checkDataset(imageData);
        audioClassifier.checkDataset(audioData);
        await imageClassifier.importDataset(imageData);
        await audioClassifier.importDataset(audioData);
    }

    /**
     * 学習・分類の入力元を切り替える。
     * マイクから切り替えたときはマイクを停止する。カメラの表示はビデオブロックに任せる。
     * @param {object} args - the block's arguments.
     * @param {string} args.INPUT - 入力元 (InputMode)
     */
    setInputMode (args) {
        const mode = args.INPUT === InputMode.MICROPHONE ? InputMode.MICROPHONE : InputMode.CAMERA;
        if (mode === this.inputMode) return;
        this.inputMode = mode;
        this.clearLabelCache();
        if (mode !== InputMode.MICROPHONE && this.audioClassifier) {
            this.audioClassifier.stopMicrophone();
        }
    }

    /**
     * A scratch command block handle that configures the video state from
     * passed arguments.
     * @param {object} args - the block arguments
     * @param {VideoState} args.VIDEO_STATE - the video state to set the device to
     */
    videoToggle (args) {
        const state = args.VIDEO_STATE;
        if (state === VideoState.OFF) {
            this.runtime.ioDevices.video.disableVideo();
        } else {
            this.runtime.ioDevices.video.enableVideo();
            // Mirror if state is ON. Do not mirror if state is ON_FLIPPED.
            this.runtime.ioDevices.video.mirror = state === VideoState.ON;
        }
    }

    /**
     * エラーメッセージをダイアログで表示する。
     * @param {string} contentHtml - 表示する内容
     * @returns {Promise} ダイアログが閉じられたときに resolve する Promise
     */
    openErrorDialog (contentHtml) {
        if (this.errorDialogOpened) {
            // prevent to open multiple dialogs
            return Promise.resolve(null);
        }
        this.errorDialogOpened = true;
        const errorDialog = document.createElement('dialog');
        errorDialog.style.padding = '0px';
        errorDialog.style.position = 'relative';

        // Close button
        const closeButton = document.createElement('button');
        closeButton.textContent = '✕';
        closeButton.style.position = 'absolute';
        closeButton.style.top = '8px';
        closeButton.style.right = '8px';
        closeButton.style.border = 'none';
        closeButton.style.background = 'none';
        closeButton.style.fontSize = '24px';
        closeButton.style.cursor = 'pointer';
        closeButton.style.outline = 'none';
        closeButton.style.display = 'flex';
        closeButton.style.justifyContent = 'center';
        closeButton.style.alignItems = 'center';
        errorDialog.appendChild(closeButton);

        const dialogFace = document.createElement('div');
        dialogFace.style.padding = '32px';
        dialogFace.innerHTML = contentHtml;
        errorDialog.appendChild(dialogFace);

        return new Promise(resolve => {
            const close = () => {
                resolve(false);
            };
            closeButton.onclick = close;
            errorDialog.addEventListener('keydown', e => {
                if (e.code === 'Escape') {
                    close();
                }
            });

            document.body.appendChild(errorDialog);
            errorDialog.showModal();
            closeButton.focus();
        })
            .finally(() => {
                document.body.removeChild(errorDialog);
                this.errorDialogOpened = false;
            });
    }

    /**
     * @returns {object} metadata for this extension and its blocks.
     */
    getInfo () {
        setupTranslations();
        return {
            id: ExtensionBlocks.EXTENSION_ID,
            name: ExtensionBlocks.EXTENSION_NAME,
            extensionURL: ExtensionBlocks.extensionURL,
            blockIconURI: blockIcon,
            showStatusButton: false,
            blocks: [
                {
                    opcode: 'mlTrainLabel1',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'xcxml.trainLabel1',
                        default: 'train label 1',
                        description: 'train the current camera image as label 1'
                    })
                },
                {
                    opcode: 'mlTrainLabel2',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'xcxml.trainLabel2',
                        default: 'train label 2',
                        description: 'train the current camera image as label 2'
                    })
                },
                {
                    opcode: 'mlTrainLabel3',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'xcxml.trainLabel3',
                        default: 'train label 3',
                        description: 'train the current camera image as label 3'
                    })
                },
                {
                    opcode: 'mlLabel',
                    blockType: BlockType.REPORTER,
                    disableMonitor: false,
                    text: formatMessage({
                        id: 'xcxml.label',
                        default: 'label',
                        description: 'the label recognized from the current camera image'
                    })
                },
                {
                    opcode: 'mlReset',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'xcxml.reset',
                        default: 'reset learning',
                        description: 'reset all trained examples'
                    })
                },
                {
                    opcode: 'mlSaveData',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'xcxml.saveData',
                        default: 'save learning data',
                        description: 'save trained examples as a file'
                    })
                },
                {
                    opcode: 'mlLoadData',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'xcxml.loadData',
                        default: 'load learning data',
                        description: 'load trained examples from a file'
                    })
                },
                '---',
                {
                    opcode: 'setInputMode',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'xcxml.setInput',
                        default: 'set input to [INPUT]',
                        description: 'select camera or microphone as the input for learning and recognition'
                    }),
                    arguments: {
                        INPUT: {
                            type: ArgumentType.STRING,
                            menu: 'INPUT',
                            defaultValue: InputMode.CAMERA
                        }
                    }
                },
                {
                    opcode: 'videoToggle',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'videoSensing.videoToggle',
                        default: 'turn video [VIDEO_STATE]',
                        description: 'Controls display of the video preview layer'
                    }),
                    arguments: {
                        VIDEO_STATE: {
                            type: ArgumentType.STRING,
                            menu: 'VIDEO_STATE',
                            defaultValue: VideoState.ON
                        }
                    }
                }
            ],
            menus: {
                INPUT: {
                    acceptReporters: true,
                    items: [
                        {
                            text: formatMessage({
                                id: 'xcxml.input.camera',
                                default: 'camera',
                                description: 'camera option of the "set input to [INPUT]" block'
                            }),
                            value: InputMode.CAMERA
                        },
                        {
                            text: formatMessage({
                                id: 'xcxml.input.microphone',
                                default: 'microphone',
                                description: 'microphone option of the "set input to [INPUT]" block'
                            }),
                            value: InputMode.MICROPHONE
                        }
                    ]
                },
                VIDEO_STATE: {
                    acceptReporters: true,
                    items: [
                        {
                            text: formatMessage({
                                id: 'videoSensing.off',
                                default: 'off',
                                description: 'Option for the "turn video [STATE]" block'
                            }),
                            value: VideoState.OFF
                        },
                        {
                            text: formatMessage({
                                id: 'videoSensing.on',
                                default: 'on',
                                description: 'Option for the "turn video [STATE]" block'
                            }),
                            value: VideoState.ON
                        },
                        {
                            text: formatMessage({
                                id: 'videoSensing.onFlipped',
                                default: 'on flipped',
                                description: 'Option for the "turn video [STATE]" block that causes the video to be flipped' +
                                    ' horizontally (reversed as in a mirror)'
                            }),
                            value: VideoState.ON_FLIPPED
                        }
                    ]
                }
            }
        };
    }
}

export {ExtensionBlocks as default, ExtensionBlocks as blockClass};
