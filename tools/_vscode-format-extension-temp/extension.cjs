const vscode = require("vscode");
const fs = require("node:fs/promises");
const crypto = require("node:crypto");
const path = require("node:path");

async function activate(context) {
    const workDir = process.env.SRG_FORMAT_EVIDENCE;
    await fs.mkdir(workDir, {recursive: true});
    try {
        const uri = vscode.Uri.file(process.env.SRG_FORMAT_FILE);
        const document = await vscode.workspace.openTextDocument(uri);
        await vscode.window.showTextDocument(document, {preview: false});
        const formatter = vscode.extensions.getExtension("vscode.html-language-features");
        if (!formatter) throw new Error("VS Code builtin vscode.html-language-features not found");
        await formatter.activate();
        const selected = vscode.workspace.getConfiguration("editor", uri).get("defaultFormatter");
        if (selected !== "vscode.html-language-features") {
            throw new Error("Wrong default formatter: " + selected);
        }
        const before = document.getText();
        await vscode.commands.executeCommand("editor.action.formatDocument");
        await document.save();
        const after = document.getText();
        const output = path.join(workDir, "revgeo.step2.vscode-formatted.html");
        await fs.writeFile(output, after, "utf8");
        const report = {
            editor: "Visual Studio Code",
            version: vscode.version,
            extensionId: formatter.id,
            extensionVersion: formatter.packageJSON.version,
            extensionActivated: formatter.isActive,
            operation: "editor.action.formatDocument",
            editorDefaultFormatter: selected,
            inputSha256: crypto.createHash("sha256").update(before).digest("hex"),
            outputSha256: crypto.createHash("sha256").update(after).digest("hex"),
            inputBytes: Buffer.byteLength(before),
            outputBytes: Buffer.byteLength(after),
            formattedFile: output,
            documentSaved: !document.isDirty
        };
        if (report.inputSha256 !== process.env.SRG_FORMAT_INPUT_SHA) {
            throw new Error("Unexpected formatted input bytes: " + report.inputSha256);
        }
        if (!report.extensionActivated || !report.documentSaved) throw new Error("Formatter not activated or output unsaved");
        await fs.writeFile(path.join(workDir, "vscode-format-report.json"), JSON.stringify(report, null, 2) + "\n");
    } catch (error) {
        await fs.writeFile(path.join(workDir, "vscode-format-error.txt"), String(error?.stack || error) + "\n");
    } finally {
        await vscode.commands.executeCommand("workbench.action.quit");
    }
}
module.exports = {activate};
