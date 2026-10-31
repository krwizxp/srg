import {readFileSync,createReadStream,statSync} from "node:fs";
import {createServer} from "node:http";
import {resolve,sep} from "node:path";
import {chromium,firefox,webkit} from "playwright";

const expected=readFileSync("validation/probe-output.txt","utf8").trim().split(/\r?\n/);
if(expected.length!==750)throw Error("native Rust probe has wrong sample count");
const root=process.cwd();
const server=createServer((req,res)=>{
    let path=new URL(req.url,"http://127.0.0.1").pathname;
    if(path==="/")path="/revgeo.html";
    const file=resolve(root,"."+path);
    if(!file.startsWith(root+sep)){res.writeHead(403);res.end();return;}
    try{
        const stat=statSync(file);
        if(!stat.isFile())throw Error("directory");
        res.writeHead(200,{"Cache-Control":"no-store","Content-Length":stat.size,"Content-Type":path.endsWith(".js")?"text/javascript":path.endsWith(".html")?"text/html":"application/octet-stream"});
        createReadStream(file).pipe(res);
    }catch{res.writeHead(404);res.end();}
});
await new Promise(ok=>server.listen(0,"127.0.0.1",ok));
const origin="http://127.0.0.1:"+server.address().port+"/";
try{
    for(const [name,engine] of [["chromium",chromium],["firefox",firefox],["webkit",webkit]]){
        const browser=await engine.launch({headless:true});
        try{
            const page=await browser.newPage();
            await page.goto(origin+"revgeo.html",{waitUntil:"domcontentloaded"});
            const actual=await page.evaluate(async()=>{
                const module=await import("./korea_boundary.js?rust-parity=1");
                const mask=(1n<<64n)-1n;
                const increment=0x9e3779b97f4a7c15n;
                function splitmix(s) {
                    s.value=(s.value+increment)&mask;
                    let x=s.value;
                    x=((x^(x>>30n))*0xbf58476d1ce4e5b9n)&mask;
                    x=((x^(x>>27n))*0x94d049bb133111ebn)&mask;
                    return (x^(x>>31n))&mask;
                }
                const state={value:0x123456789abcdef0n};
                const bits=new DataView(new ArrayBuffer(8));
                const toHex=n=>{bits.setFloat64(0,n,false);return bits.getBigUint64(0,false).toString(16).padStart(16,"0");};
                const results=[];
                for(let i=0;i<750;i++){
                    const word=splitmix(state);
                    const supplemental={value:word^0xa5a5a5a5a5a5a5a5n};
                    const iterator=module.sampleKorea(word);
                    let result=await iterator.next(),reads=0;
                    while(!result.done&&reads<1024){
                        reads++;
                        result=await iterator.next({value:splitmix(supplemental),bitsRemaining:64});
                    }
                    if(!result.done)throw Error("random retries exhausted");
                    const [lat,lon]=result.value;
                    results.push(toHex(lat)+" "+toHex(lon)+" "+reads);
                }
                return results;
            });
            let different=0;
            for(let i=0;i<750;i++){if(actual[i]!==expected[i]){different++;if(different<5)console.log("RUST_WEB_MISMATCH "+JSON.stringify({name,i,expected:expected[i],actual:actual[i]}));}}
            console.log("RUST_WEB "+JSON.stringify({browser:name,version:browser.version(),total:750,different}));
            if(different)throw Error("Rust and JavaScript production polygon sampler disagree");
        }finally{await browser.close();}
    }
}finally{await new Promise(ok=>server.close(ok));}
