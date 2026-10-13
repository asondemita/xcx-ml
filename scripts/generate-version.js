/**
 * package.json の version を拡張機能のソースに書き出す。
 * ビルド前 (prebuild / prestart) に実行され、生成した version.js は
 * 拡張機能ライブラリのカード名に「(vX.Y.Z)」として表示される。
 */
const fs = require('fs');
const path = require('path');

const packageJson = require('../package.json');
const outputPath = path.join(__dirname, '..', 'src', 'gui', 'lib', 'libraries', 'extensions', 'entry', 'version.js');

const content = `// このファイルは scripts/generate-version.js が package.json から生成する。直接編集しないこと。
const version = '${packageJson.version}';
export default version;
`;

fs.writeFileSync(outputPath, content);
process.stdout.write(`version.js generated: v${packageJson.version}\n`);
