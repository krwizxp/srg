const path = require("node:path");
const { runTests } = require("@vscode/test-electron");
runTests({
    extensionDevelopmentPath: path.resolve("validation/vscode_fmt"),
    extensionTestsPath: path.resolve("validation/vscode_fmt/format_test.cjs"),
    launchArgs: ["--no-sandbox", "--disable-gpu", "--disable-workspace-trust", "--skip-welcome"],
}).catch(error => {
    console.error(error);
    process.exitCode = 1;
});
