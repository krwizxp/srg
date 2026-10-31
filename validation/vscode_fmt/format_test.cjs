const vscode = require("vscode");
const fs = require("node:fs");
const path = require("node:path");
module.exports = async () => {
    const htmlPath = path.resolve("revgeo.html");
    const document = await vscode.workspace.openTextDocument(vscode.Uri.file(htmlPath));
    await vscode.window.showTextDocument(document, { preview: false });
    const formatter = vscode.extensions.getExtension("vscode.html-language-features");
    if (!formatter) throw new Error("The built-in VS Code HTML formatter is unavailable");
    await formatter.activate();
    await vscode.workspace.getConfiguration("editor", document.uri)
        .update("defaultFormatter", "vscode.html-language-features", vscode.ConfigurationTarget.Global);
    await vscode.workspace.getConfiguration("html.format", document.uri)
        .update("contentUnformatted", "script,style,svg", vscode.ConfigurationTarget.Global);
    await vscode.commands.executeCommand("editor.action.formatDocument");
    if (!await document.save()) throw new Error("VS Code failed to save the formatted HTML");
    const finalHtml = fs.readFileSync(htmlPath, "utf8");
    if (!finalHtml.includes('import { convertParallel } from "./srg_parallel.js?v=1";'))
        throw new Error("The VS Code formatting pass lost the Worker integration");
    console.log("VSCODE_HTML_FORMAT_SAVED "+JSON.stringify({bytes:finalHtml.length,extension:formatter.id,version:formatter.packageJSON.version}));
};
