let ctxPad, ctxThumb, drawing=false, brush=14;
let model=null;

window.addEventListener("DOMContentLoaded",()=>{
  const pad=document.getElementById("pad"); ctxPad=pad.getContext("2d");
  ctxPad.fillStyle="white"; ctxPad.fillRect(0,0,pad.width,pad.height);
  const thumb=document.getElementById("thumb"); ctxThumb=thumb.getContext("2d");

  fetch("models/mlp_p1.json?v=1",{cache:"no-store"})
    .then(r=>r.json()).then(js=>loadModel(js)).catch(e=>{
      document.getElementById("status").textContent="Model load failed"; console.error(e);
    });

  pad.addEventListener("mousedown",e=>{drawing=true; draw(e)});
  pad.addEventListener("mouseup",()=>drawing=false);
  pad.addEventListener("mouseout",()=>drawing=false);
  pad.addEventListener("mousemove",e=>{if(drawing) draw(e)});
  pad.addEventListener("touchstart",e=>{drawing=true; draw(e.touches[0]); e.preventDefault();});
  pad.addEventListener("touchmove",e=>{if(drawing) draw(e.touches[0]); e.preventDefault();});
  pad.addEventListener("touchend",()=>drawing=false);

  document.getElementById("brush").oninput=e=>brush=+e.target.value;
  document.getElementById("clearBtn").onclick=clearPad;
  document.getElementById("predictBtn").onclick=predict;
  document.getElementById("downloadCsvBtn").onclick=downloadCsv;
  window.addEventListener("keydown",e=>{
    if(e.code==="Space"){e.preventDefault();predict();}
    if(e.key==="c"||e.key==="C"){clearPad();}
  });
});

function loadModel(js){
  function num(x){x=Number(x); return Number.isFinite(x)?x:0;}
  model={...js};
  ["W1","b1","W2","b2","mu"].forEach(k=>model[k]=js[k].map(num));
  model.W1=reshape(model.W1,[128,784]);
  model.W2=reshape(model.W2,[10,128]);
  document.getElementById("status").textContent="Model loaded.";
}

function clearPad(){
  ctxPad.fillStyle="white"; ctxPad.fillRect(0,0,280,280);
  document.getElementById("predVal").textContent="—";
  ctxThumb.clearRect(0,0,28,28);
  document.getElementById("scores").textContent="";
}

function draw(e){
  const r=pad.getBoundingClientRect();
  const x=e.clientX-r.left, y=e.clientY-r.top;
  ctxPad.fillStyle="black";
  ctxPad.beginPath(); ctxPad.arc(x,y,brush,0,2*Math.PI); ctxPad.fill();
}

function downsample(){
  const img=ctxPad.getImageData(0,0,280,280).data;
  const out=new Float32Array(784);
  for(let y=0;y<28;y++)for(let x=0;x<28;x++){
    let sum=0;
    for(let dy=0;dy<10;dy++)for(let dx=0;dx<10;dx++){
      const ix=((y*10+dy)*280+(x*10+dx))*4;
      sum+=img[ix]; // red channel, 0 black, 255 white
    }
    let v=1-(sum/100/255); // invert: black ink → 1
    out[y*28+x]=v;
    ctxThumb.fillStyle=`rgb(${255*(1-v)},${255*(1-v)},${255*(1-v)})`;
    ctxThumb.fillRect(x,y,1,1);
  }
  return out;
}

function predict(){
  if(!model){alert("Model not loaded");return;}
  const x=downsample();
  if(document.getElementById("centerChk").checked && model.mu)
    for(let i=0;i<784;i++) x[i]-=model.mu[i];
  const h=new Float32Array(128);
  for(let i=0;i<128;i++){
    let s=model.b1[i]; for(let j=0;j<784;j++) s+=model.W1[i][j]*x[j];
    h[i]=Math.max(0,s);
  }
  const logits=new Float32Array(10);
  for(let i=0;i<10;i++){
    let s=model.b2[i]; for(let j=0;j<128;j++) s+=model.W2[i][j]*h[j];
    logits[i]=s;
  }
  const probs=softmax(logits);
  const top=[...probs.map((p,i)=>[i,p])].sort((a,b)=>b[1]-a[1]).slice(0,3);
  document.getElementById("predVal").textContent=top[0][0]+" ("+(top[0][1]*100).toFixed(1)+"%)";
  document.getElementById("scores").textContent=probs.map((p,i)=>i+": "+p.toFixed(3)).join("\n");
}

function softmax(arr){
  const m=Math.max(...arr); const ex=arr.map(v=>Math.exp(v-m));
  const s=ex.reduce((a,b)=>a+b,0);
  return ex.map(v=>v/s);
}
function reshape(flat,shape){
  const [r,c]=shape, out=[]; for(let i=0;i<r;i++) out.push(flat.slice(i*c,(i+1)*c));
  return out;
}
function downloadCsv(){
  const x=downsample();
  let csv=""; for(let i=0;i<28;i++) csv+=x.slice(i*28,(i+1)*28).join(",")+"\n";
  const blob=new Blob([csv],{type:"text/csv"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob);
  a.download="digit.csv"; a.click();
}
