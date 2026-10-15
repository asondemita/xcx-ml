/**
 * This is an extension for Xcratch.
 */

import iconURL from './entry-icon.png';
import insetIconURL from './inset-icon.png';
import translations from './translations.json';
import version from './version';

/**
 * Formatter to translate the messages in this extension.
 * This will be replaced which is used in the React component.
 * @param {object} messageData - data for format-message
 * @returns {string} - translated message for the current locale
 */
let formatMessage = messageData => messageData.defaultMessage;

const entry = {
    get name () {
        return formatMessage({
            defaultMessage: 'Machine Learning',
            description: 'Name for the "Machine Learning" extension',
            id: 'xcxml.entry.name'
        });
    },
    extensionId: 'xcxml',
    extensionURL: 'https://asondemita.github.io/xcx-ml/dist/xcx-ml.mjs',
    collaborator: 'asondemita',
    iconURL: iconURL,
    insetIconURL: insetIconURL,
    get description () {
        // 説明文の末尾にバージョンを付けて、利用者が新旧を見分けられるようにする
        return `${formatMessage({
            defaultMessage: 'Recognize camera images and microphone sounds with machine learning (KNN).',
            description: 'Description for the "Machine Learning" extension',
            id: 'xcxml.entry.description'
        })} (v${version})`;
    },
    featured: true,
    disabled: false,
    bluetoothRequired: false,
    internetConnectionRequired: true,
    helpLink: 'https://github.com/asondemita/xcx-ml/',
    setFormatMessage: formatter => {
        formatMessage = formatter;
    },
    translationMap: translations,
    tags: ['others']
};

export {entry}; // loadable-extension needs this line.
export default entry;
