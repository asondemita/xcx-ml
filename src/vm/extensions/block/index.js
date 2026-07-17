import BlockType from '../../extension-support/block-type';
import ArgumentType from '../../extension-support/argument-type';
import translations from './translations.json';
import blockIcon from './block-icon.png';

import ImageClassifier from './image-classifier';

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
let extensionURL = 'https://asondemita.github.io/xcx-ml/dist/xcxml.mjs';

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
 * Scratch 3.0 blocks for machine learning (KNN image classification).
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
     * 現在のカメラ画像を指定ラベルの例として学習する。
     * @param {string} label - ラベル
     * @returns {Promise} 学習完了で resolve する Promise
     */
    async mlTrain (label) {
        try {
            const input = await this.getVideoInput();
            await this.getImageClassifier().train(input, label);
        } catch (error) {
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
     * 現在のカメラ画像を分類してラベルを返す。
     * 未学習のときはカメラを起動せずに空文字列を返す。
     * 短時間に連続で呼ばれた場合 (ステージのモニターなど) はキャッシュを返し、
     * 推論が過剰に走らないようにする。
     * @returns {string|Promise<string>} 分類されたラベル
     */
    mlLabel () {
        const classifier = this.imageClassifier;
        if (!classifier || !classifier.hasExamples()) return '';
        if ((Date.now() - this.mlLabelUpdatedTime) < this.mlLabelUpdateIntervalTime) {
            return this.mlLastLabel;
        }
        if (!this.mlClassifyRequest) {
            this.mlClassifyRequest = this.getVideoInput()
                .then(input => classifier.classify(input))
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
     * 学習した内容をすべてリセットする。
     */
    mlReset () {
        if (this.imageClassifier) {
            this.imageClassifier.reset();
        }
        this.mlLastLabel = '';
        this.mlLabelUpdatedTime = 0;
    }

    /**
     * 学習データを JSON ファイルとして保存 (ダウンロード) する。
     */
    mlSaveData () {
        const classifier = this.imageClassifier;
        if (!classifier || !classifier.hasExamples()) {
            // 学習データが無いときは何もしない
            return;
        }
        const json = classifier.serialize();
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
            const parsed = JSON.parse(await file.text());
            await this.getImageClassifier().restore(parsed);
            this.mlLastLabel = '';
            this.mlLabelUpdatedTime = 0;
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
