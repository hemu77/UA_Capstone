// Transport compression changes no recorded bytes: verify after decompression.
export async function decodeVerifiedEvents(input,expectedHash){
  let bytes=new Uint8Array(input);
  if(bytes[0]===0x1f&&bytes[1]===0x8b){
    if(typeof DecompressionStream==='undefined')throw Error('This browser cannot decompress saved replay. Use a current browser; final graphs remain available.');
    const reader=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip')).getReader();
    const chunks=[];let size=0;
    try{
      for(;;){
        const {done,value}=await reader.read();if(done)break;
        size+=value.length;if(size>2_000_000)throw Error('Saved replay exceeds its size limit');
        chunks.push(value);
      }
    }finally{await reader.cancel();}
    bytes=new Uint8Array(size);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}
  }
  if(bytes.length>2_000_000)throw Error('Saved replay exceeds its size limit');
  const digest=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(b=>b.toString(16).padStart(2,'0')).join('');
  if(digest!==expectedHash)throw Error('Recorded replay failed its integrity check');
  return JSON.parse(new TextDecoder().decode(bytes));
}
