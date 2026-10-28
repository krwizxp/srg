const MAX_U64 = (1n << 64n) - 1n;
const MAX_PAGE = 15876;
const DEGREES = 0.017453292519943295;
const ECCENTRICITY_SQUARED = 0.006694380022900787;
const SURFACE_SCALE = 12309072078.888908;
let boundaryPromise;
let preparation;
let requestedPreparation;
const parts = new Map();
const pages = new Map();
const indexes = new Map();
function productError(a,b,product) {
    const ca=134217729*a, ah=ca-(ca-a), al=a-ah;
    const cb=134217729*b, bh=cb-(cb-b), bl=b-bh;
    return ((ah*bh-product)+ah*bl+al*bh)+al*bl;
}
function determinant(a,b,c,d) {
    const left=a*d, right=b*c;
    let expansion=[];
    for(let value of [productError(a,d,left),left,-productError(b,c,right),-right]) {
        const result=[];
        for(const entry of expansion) {
            const total=value+entry, virtual=total-value;
            const error=(value-(total-virtual))+(entry-virtual);
            if(error!==0)result.push(error);
            value=total;
        }
        if(value!==0||!result.length)result.push(value);
        expansion=result;
    }
    let value=0;
    for(const entry of expansion)value+=entry;
    return value;
}
function density(latitude) {
    const delta=(latitude-36)*DEGREES, square=delta*delta;
    const sineDelta=delta*(1+square*(-0.16666666666666666+square*(0.008333333333333333+square*(-0.0001984126984126984+square*(2.7557319223985893e-6+square*-2.505210838544172e-8)))));
    const cosineDelta=1+square*(-0.5+square*(0.041666666666666664+square*(-0.001388888888888889+square*(2.48015873015873e-5+square*(-2.755731922398589e-7+square*2.08767569878681e-9)))));
    const sine=0.5877852522924731*cosineDelta+0.8090169943749475*sineDelta;
    const cosine=0.8090169943749475*cosineDelta-0.5877852522924731*sineDelta;
    const denominator=1-ECCENTRICITY_SQUARED*sine*sine;
    return cosine/(denominator*denominator);
}
function unit(word) { return (Number(word>>12n)+0.5)/4503599627370496; }
function propose(vertices,weight,words) {
    const [a,b,c]=vertices;
    const det=determinant(b[1]-a[1],b[0]-a[0],c[1]-a[1],c[0]-a[0]);
    const area=Math.abs(det)*0.5;
    if(!(area>0&&Number.isFinite(area)&&weight>0n))throw new Error("대한민국 삼각형 면적 오류");
    let u=unit(words[0]),v=unit(words[1]);
    if(u+v>1){u=1-u;v=1-v;}
    const latitude=(a[0]+u*(b[0]-a[0]))+v*(c[0]-a[0]);
    const longitude=(a[1]+u*(b[1]-a[1]))+v*(c[1]-a[1]);
    for(const [start,end]of[[a,b],[b,c],[c,a]]) {
        const side=determinant(end[1]-start[1],end[0]-start[0],longitude-start[1],latitude-start[0]);
        if(det>0&&side<0||det<0&&side>0)return null;
    }
    const probability=((area*SURFACE_SCALE)*density(latitude))*1000000/Number(weight);
    if(!(probability>0&&probability<=1&&Number.isFinite(probability)))throw new Error("대한민국 면적 수락 확률 오류");
    return unit(words[2])<probability?[latitude,longitude]:null;
}
async function boundedBytes(response,maximum) {
    if(!response.body)throw new Error("대한민국 경계 응답 본문 오류");
    const reader=response.body.getReader(),chunks=[];let size=0;
    try {
        for(;;) {
            const {value,done}=await reader.read();if(done)break;
            size+=value.length;if(size>maximum)throw new Error("대한민국 경계 데이터 크기 한도를 초과했습니다.");
            chunks.push(value);
        }
    }catch(error){await reader.cancel().catch(()=>{});throw error;}
    finally{reader.releaseLock();}
    const bytes=new Uint8Array(size);let offset=0;
    for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}
    return bytes;
}
function search(data,start,count,value) {
    let low=0,high=count;
    while(low<high){const middle=Math.floor((low+high)/2);if(data.getBigUint64(start+middle*16,true)<value)low=middle+1;else high=middle;}
    if(low>=count)throw new Error("대한민국 경계 데이터 범위 오류");
    return low;
}
async function loadBoundary() {
    if(!boundaryPromise)boundaryPromise=(async()=>{
        const response=await fetch(new URL("./data/korea-boundary-v2/index.bin",import.meta.url));
        if(!response.ok)throw new Error('대한민국 경계 데이터 불러오기 실패: HTTP '+response.status);
        const bytes=await boundedBytes(response,65536),data=new DataView(bytes.buffer);
        if(bytes.length<32||new TextDecoder().decode(bytes.subarray(0,8))!=="SRGKR002")throw new Error("대한민국 경계 데이터 버전 오류");
        const count=data.getUint32(8,true),triangles=data.getUint32(12,true),total=data.getBigUint64(24,true);
        if(!count||32+count*16!==bytes.length||!triangles||triangles>20000000||data.getUint32(16,true)!==14||data.getUint32(20,true)!==256||!total)throw new Error("대한민국 경계 인덱스 오류");
        let previous=-1n,blocks=0;const firstBlocks=[];
        for(let i=0;i<count;i++){
            const entry=32+i*16,end=data.getBigUint64(entry,true),size=data.getUint32(entry+8,true),n=data.getUint32(entry+12,true);
            if(end<=previous||!n||size>524288||size<=16+n*16)throw new Error("대한민국 경계 조각 인덱스 오류");
            firstBlocks.push(blocks);blocks+=n;previous=end;
        }
        if(previous!==total-1n||blocks!==Math.ceil(triangles/256))throw new Error("대한민국 경계 가중치 범위 오류");
        return {bytes,data,count,total,triangles,firstBlocks};
    })().catch(error=>{boundaryPromise=null;throw error});
    return boundaryPromise;
}
async function loadPart(boundary,number) {
    if(parts.has(number)){const value=parts.get(number);parts.delete(number);parts.set(number,value);return value;}
    const pending=(async()=>{
        const name=String(number).padStart(3,"0"),response=await fetch(new URL("./data/korea-boundary-v2/part-"+name+".bin",import.meta.url));
        if(!response.ok)throw new Error('대한민국 경계 블록 불러오기 실패: HTTP '+response.status);
        const bytes=await boundedBytes(response,524288),data=new DataView(bytes.buffer),root=32+number*16;
        if(bytes.length<16||bytes.length!==boundary.data.getUint32(root+8,true)||new TextDecoder().decode(bytes.subarray(0,8))!=="SRGKP002")throw new Error("대한민국 경계 조각 구조 오류");
        const count=data.getUint32(8,true),first=data.getUint32(12,true);
        if(count!==boundary.data.getUint32(root+12,true)||first!==boundary.firstBlocks[number]||16+count*16>=bytes.length)throw new Error("대한민국 경계 블록 개수 오류");
        let previous=number?boundary.data.getBigUint64(root-16,true):-1n,offset=16+count*16;
        for(let i=0;i<count;i++){
            const entry=16+i*16,end=data.getBigUint64(entry,true),start=data.getUint32(entry+8,true),length=data.getUint32(entry+12,true);
            if(end<=previous||start!==offset||!length||start+length>bytes.length)throw new Error("대한민국 경계 블록 인덱스 오류");
            previous=end;offset=start+length;
        }
        if(previous!==boundary.data.getBigUint64(root,true)||offset!==bytes.length)throw new Error("대한민국 경계 조각 범위 오류");
        const metadata={data:new DataView(bytes.slice(0,16+count*16).buffer),count,first,number};
        indexes.set(number,metadata);
        return {bytes,...metadata};
    })();
    parts.set(number,pending);if(parts.size>8)parts.delete(parts.keys().next().value);
    try{return await pending}catch(error){if(parts.get(number)===pending)parts.delete(number);throw error;}
}
async function loadPage(boundary,part,local) {
    const block=part.first+local;
    if(pages.has(block)){const value=pages.get(block);pages.delete(block);pages.set(block,value);return value;}
    const pending=(async()=>{
        const entry=16+local*16,start=part.data.getUint32(entry+8,true),length=part.data.getUint32(entry+12,true);
        const payload=part.bytes??(await loadPart(boundary,part.number)).bytes;
        const stream=new Blob([payload.subarray(start,start+length)]).stream().pipeThrough(new DecompressionStream("deflate"));
        const shuffled=await boundedBytes(new Response(stream),MAX_PAGE);
        if(shuffled.length<4)throw new Error("대한민국 경계 블록 길이 오류");
        const bytes=new Uint8Array(shuffled.length);bytes.set(shuffled.subarray(0,4));const data=new DataView(bytes.buffer);
        const vertices=data.getUint16(0,true),count=data.getUint16(2,true),records=4+vertices*16;
        const expected=Math.min(256,boundary.triangles-block*256);
        if(!vertices||vertices>768||count!==expected||bytes.length!==records+count*14)throw new Error("대한민국 경계 블록 오류");
        for(let word=0;word<vertices*2;word++)for(let lane=0;lane<8;lane++)bytes[4+word*8+lane]=shuffled[4+lane*vertices*2+word];
        for(let row=0;row<count;row++)for(let lane=0;lane<14;lane++)bytes[records+row*14+lane]=shuffled[records+lane*count+row];
        const previous=local?part.data.getBigUint64(entry-16,true):part.number?boundary.data.getBigUint64(32+(part.number-1)*16,true):-1n;
        let end=previous;
        for(let i=0;i<count;i++){
            const record=records+i*14,width=data.getBigUint64(record,true);
            if(!width)throw new Error("대한민국 경계 가중치 오류");
            end+=width;if(end>=boundary.total)throw new Error("대한민국 경계 가중치 한도 오류");data.setBigUint64(record,end,true);
            for(let corner=0;corner<3;corner++)if(data.getUint16(record+8+corner*2,true)>=vertices)throw new Error("대한민국 경계 정점 오류");
        }
        if(end!==part.data.getBigUint64(entry,true))throw new Error("대한민국 경계 누적 가중치 오류");
        for(let i=0;i<vertices;i++){const lat=data.getFloat64(4+i*16,true),lon=data.getFloat64(12+i*16,true);if(!(lat>=33&&lat<39&&lon>=124&&lon<132))throw new Error("대한민국 경계 정점 범위 오류");}
        return {data,records,count,previous};
    })();
    pages.set(block,pending);if(pages.size>256)pages.delete(pages.keys().next().value);
    try{return await pending}catch(error){if(pages.get(block)===pending)pages.delete(block);parts.delete(part.number);indexes.delete(part.number);throw error;}
}
async function selected(boundary,value) {
    const number=search(boundary.data,32,boundary.count,value),part=indexes.get(number)??await loadPart(boundary,number);
    const local=search(part.data,16,part.count,value),page=await loadPage(boundary,part,local);
    let low=0,high=page.count;
    while(low<high){const middle=Math.floor((low+high)/2);if(page.data.getBigUint64(page.records+middle*14,true)<value)low=middle+1;else high=middle;}
    if(low>=page.count)throw new Error("대한민국 경계 삼각형 범위 오류");
    const record=page.records+low*14,end=page.data.getBigUint64(record,true);
    const begin=low?page.data.getBigUint64(record-14,true)+1n:page.previous+1n;
    const vertices=[];
    for(let corner=0;corner<3;corner++){const v=page.data.getUint16(record+8+corner*2,true);vertices.push([page.data.getFloat64(4+v*16,true),page.data.getFloat64(12+v*16,true)]);}
    return {weight:end-begin+1n,vertices};
}
function entropy(request) {
    const value=request?.value;
    if(typeof value!=="bigint"||value<0n||value>MAX_U64)throw new RangeError("64비트 보완 난수 범위를 벗어났습니다.");
    return value;
}
export async function* sampleKorea(value) {
    if(typeof value!=="bigint"||value<0n||value>MAX_U64)throw new RangeError("64비트 난수 범위를 벗어났습니다.");
    const boundary=await loadBoundary(),limit=MAX_U64-(boundary.total-1n);
    for(let attempt=0;attempt<1024;attempt++){
        const position=value%boundary.total;
        if(value-position>limit){value=entropy(yield);continue;}
        const triangle=await selected(boundary,position);
        const words=[entropy(yield),entropy(yield),entropy(yield)];
        const point=propose(triangle.vertices,triangle.weight,words);
        if(point)return point;
        value=entropy(yield);
    }
    throw new Error("대한민국 균등 좌표 추출 시도 한도를 초과했습니다.");
}
export function prepareKorea(value) {
    requestedPreparation=value;
    if(!preparation)preparation=(async()=>{
        let current;
        do{
            current=requestedPreparation;
            const boundary=await loadBoundary();
            if(typeof current==="bigint"&&current>=0n&&current<=MAX_U64){
                const position=current%boundary.total,limit=MAX_U64-(boundary.total-1n);
                if(current-position<=limit)await selected(boundary,position);
            }
        }while(current!==requestedPreparation);
    })().finally(()=>{preparation=null});
    return preparation;
}
