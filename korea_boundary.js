const MAX_U64 = (1n << 64n) - 1n;
const MAX_PAGE = 15876;
const DEGREES = 0.017453292519943295;
const ECCENTRICITY_SQUARED = 0.006694380022900787;
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
const QUADRATURE = [
    [0.06943184420297371,0.17392742256872692],
    [0.33000947820757187,0.32607257743127305],
    [0.6699905217924281,0.32607257743127305],
    [0.9305681557970262,0.17392742256872692]
];
function triangularDensity(start,delta) {
    let total=0;
    for(const [position,weight] of QUADRATURE)total+=(2*position*weight)*density(start+delta*position);
    return total;
}
function reverse32(value) {
    value=(value>>>1&0x55555555)|(value&0x55555555)<<1;
    value=(value>>>2&0x33333333)|(value&0x33333333)<<2;
    value=(value>>>4&0x0F0F0F0F)|(value&0x0F0F0F0F)<<4;
    value=(value>>>8&0x00FF00FF)|(value&0x00FF00FF)<<8;
    return ((value>>>16)|(value<<16))>>>0;
}
function residualPosition(rank,count) {
    const vertical=Math.min((Number(rank)+0.5)/Number(count),1-Number.EPSILON/2);
    const upper=(reverse32(Number(rank&0xFFFFFFFFn))^0x9E3779B9)>>>0;
    const lower=(reverse32(Number(rank>>32n))^0x7F4A7C15)>>>12;
    const horizontal=(upper*1048576+lower+0.5)/4503599627370496;
    return [vertical,horizontal];
}
function inside(vertices,point,orientation) {
    const [a,b,c]=vertices;
    return [[a,b],[b,c],[c,a]].every(([start,end])=>{
        const side=determinant(end[1]-start[1],end[0]-start[0],point[1]-start[1],point[0]-start[0]);
        return orientation>0?side>=0:side<=0;
    });
}
function adjacent(value,up) {
    const buffer=new ArrayBuffer(8),data=new DataView(buffer);
    data.setFloat64(0,value);data.setBigUint64(0,data.getBigUint64(0)+(up?1n:-1n));
    return data.getFloat64(0);
}
function mapTriangle(vertices,rank,count) {
    const [a,b,c]=vertices;
    const orientation=determinant(b[1]-a[1],b[0]-a[0],c[1]-a[1],c[0]-a[0]);
    if(!(orientation!==0&&Number.isFinite(orientation)&&count>0n&&rank>=0n&&rank<count))throw new Error("대한민국 삼각형 면적 또는 입력 구간 오류");
    const [south,middle,north]=[...vertices].sort((left,right)=>left[0]-right[0]);
    const span=north[0]-south[0];
    if(!(span>0))throw new Error("대한민국 삼각형 위도 오류");
    const fraction=(middle[0]-south[0])/span;
    const crossing=[middle[0],south[1]+fraction*(north[1]-south[1])];
    const lowerMass=fraction*triangularDensity(south[0],middle[0]-south[0]);
    const upperMass=(1-fraction)*triangularDensity(north[0],middle[0]-north[0]);
    const [vertical,horizontal]=residualPosition(rank,count),target=vertical*(lowerMass+upperMass);
    const [apex,unit]=target<lowerMass?[south,target/lowerMass]:[north,1-(target-lowerMass)/upperMass];
    const delta=middle[0]-apex[0],mean=triangularDensity(apex[0],delta);
    let distance=Math.sqrt(Math.max(0,Math.min(1,unit)));
    for(let i=0;i<4;i++){
        if(distance===0)break;
        const mass=distance*distance*triangularDensity(apex[0],delta*distance);
        distance-=(mass-unit*mean)/(2*distance*density(apex[0]+delta*distance));
    }
    const edge=[apex[0]+distance*(middle[0]-apex[0]),apex[1]+distance*(middle[1]-apex[1])];
    const other=apex[1]+distance*(crossing[1]-apex[1]);
    const point=[edge[0],edge[1]+horizontal*(other-edge[1])];
    const center=[(a[0]+b[0]+c[0])/3,(a[1]+b[1]+c[1])/3];
    for(let i=0;i<8;i++){
        if(inside(vertices,point,orientation))return point;
        for(let axis=0;axis<2;axis++)point[axis]=adjacent(point[axis],point[axis]<center[axis]);
    }
    let nearest=a,best=Infinity;
    for(const vertex of vertices){const square=(vertex[0]-point[0])**2+(vertex[1]-point[1])**2;if(square<best){nearest=vertex;best=square;}}
    return [...nearest];
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
        const response=await fetch(new URL("./data/korea-boundary-v3/index.bin",import.meta.url));
        if(!response.ok)throw new Error('대한민국 경계 데이터 불러오기 실패: HTTP '+response.status);
        const bytes=await boundedBytes(response,65536),data=new DataView(bytes.buffer);
        if(bytes.length<32||new TextDecoder().decode(bytes.subarray(0,8))!=="SRGKR003")throw new Error("대한민국 경계 데이터 버전 오류");
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
        const name=String(number).padStart(3,"0"),response=await fetch(new URL("./data/korea-boundary-v3/part-"+name+".bin",import.meta.url));
        if(!response.ok)throw new Error('대한민국 경계 블록 불러오기 실패: HTTP '+response.status);
        const bytes=await boundedBytes(response,524288),data=new DataView(bytes.buffer),root=32+number*16;
        if(bytes.length<16||bytes.length!==boundary.data.getUint32(root+8,true)||new TextDecoder().decode(bytes.subarray(0,8))!=="SRGKP003")throw new Error("대한민국 경계 조각 구조 오류");
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
    return {begin,weight:end-begin+1n,vertices};
}
export async function* sampleKorea(value) {
    if(typeof value!=="bigint"||value<0n||value>MAX_U64)throw new RangeError("64비트 난수 범위를 벗어났습니다.");
    const boundary=await loadBoundary(),position=value*boundary.total>>64n;
    const triangle=await selected(boundary,position);
    const first=((triangle.begin<<64n)+boundary.total-1n)/boundary.total;
    const end=(((triangle.begin+triangle.weight)<<64n)+boundary.total-1n)/boundary.total;
    return mapTriangle(triangle.vertices,value-first,end-first);
}
export function prepareKorea(value) {
    requestedPreparation=value;
    if(!preparation)preparation=(async()=>{
        let current;
        do{
            current=requestedPreparation;
            const boundary=await loadBoundary();
            if(typeof current==="bigint"&&current>=0n&&current<=MAX_U64){
                await selected(boundary,current*boundary.total>>64n);
            }
        }while(current!==requestedPreparation);
    })().finally(()=>{preparation=null});
    return preparation;
}
