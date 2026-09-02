(function installFlowBrushSupport(global) {
  // Base Illustrative Brush footprint; signal-sRGB and the rendering kernel are unchanged.
  // NDC -> pixel displacement is diag((W-1)/2, (H-1)/2). No metric units.
  const EXTENT = 1.5;
  const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));

  function familyOf(row) {
    if(Math.floor(row.tag||0)>=2)return 2;
    return Math.max(row.sx,row.sy)/Math.max(.0001,Math.min(row.sx,row.sy))>=2.15?1:0;
  }

  function brushQ(x,y,row,settings={}) {
    const family=familyOf(row),majorX=row.sx>=row.sy;
    const longitudinal=majorX?x:y,transverse=majorX?y:x;
    const length=[.90,1,.74][family],widthBase=[.94,.82,.68][family];
    const shoulder=[.10,.17,.26][family],bias=[.02,-.025,-.08][family],bendAmount=[.025,.055,.035][family];
    const u=longitudinal/length,u2=u*u;
    const bend=bendAmount*u*(1-u2),dBend=bendAmount*(1-3*u2);
    const baseWidth=widthBase*(1-shoulder*u2+bias*u),dBase=widthBase*(-2*shoulder*u+bias);
    const start=settings.widthStart??1,end=settings.widthEnd??1;
    const taper=settings.widthTaperEnabled?clamp(row.taper??1,0,1):0;
    const progress=clamp(.5+.5*longitudinal,0,1),dp=longitudinal>-1&&longitudinal<1?.5*length:0;
    const taperFactor=start+taper*(end-start)*progress,dTaper=taper*(end-start)*dp;
    const rawWidth=baseWidth*taperFactor,width=Math.max(.0001,rawWidth);
    const dWidth=rawWidth>.0001?dBase*taperFactor+baseWidth*dTaper:0;
    const v=(transverse-bend)/width,v2=v*v;
    const dvdu=-dBend/width-v*dWidth/width;
    const u3=u*u2,u4=u2*u2,u5=u3*u2;
    const gl=((.55*4*u3+.45*6*u5)+4*v*v2*dvdu)/length,gt=4*v*v2/width;
    return {q:.55*u4+.45*u4*u2+v2*v2,g:majorX?[gl,gt]:[gt,gl]};
  }

  // Solve the first paint-core intersection on a ray, then differentiate the
  // implicit q(r)=level equation, not the discrete bisection decisions.
  function brushRadius(row,n,width,height,settings={}) {
    const D=[Math.max(1,width-1)/2,Math.max(1,height-1)/2];
    const c=Math.cos(row.theta),s=Math.sin(row.theta);
    const ex=Math.max(.0001,row.sx*EXTENT),ey=Math.max(.0001,row.sy*EXTENT);
    const axis=[(c*n[0]/D[0]+s*n[1]/D[1])/ex,(-s*n[0]/D[0]+c*n[1]/D[1])/ey];
    const level=settings.level??(1-clamp(settings.feather??.18,.01,.49));
    let lo=0,hi=2*Math.max(Math.hypot(D[0]*c,D[1]*s)*ex,Math.hypot(D[0]*s,D[1]*c)*ey);
    for(let k=0;k<48;k++){
      const mid=(lo+hi)*.5;
      if(brushQ(mid*axis[0],mid*axis[1],row,settings).q>level)hi=mid;else lo=mid;
    }
    const radius=(lo+hi)*.5,x=radius*axis[0],y=radius*axis[1];
    const {q,g}=brushQ(x,y,row,settings),slope=g[0]*axis[0]+g[1]*axis[1];
    if(!(slope>1e-10)||!Number.isFinite(slope)||Math.abs(q-level)>.001)return null;
    const logScale=[row.sx*EXTENT>.0001?g[0]*x/slope:0,row.sy*EXTENT>.0001?g[1]*y/slope:0];
    const theta=-(g[0]*y*ey/ex-g[1]*x*ex/ey)/slope;
    const direction=[-radius*(g[0]*c/ex-g[1]*s/ey)/D[0]/slope,-radius*(g[0]*s/ex+g[1]*c/ey)/D[1]/slope];
    return {radius,logScale,theta,direction,q,slope};
  }

  function supportPenalty(a,b,width,height,settings={}) {
    const D=[Math.max(1,width-1)/2,Math.max(1,height-1)/2];
    const delta=[(b.x-a.x)*D[0],(b.y-a.y)*D[1]],d=Math.hypot(...delta);
    const zero=()=>({center:[0,0],logScale:[0,0],theta:0});
    if(d<1e-5)return {energy:0,ratio:0,degenerate:true,a:zero(),b:zero()};
    const n=delta.map(x=>x/d),ra=brushRadius(a,n,width,height,settings),rb=brushRadius(b,n.map(x=>-x),width,height,settings);
    if(!ra||!rb)return {energy:0,ratio:0,invalid:true,a:zero(),b:zero()};
    const sum=Math.max(1e-5,ra.radius+rb.radius),ratio=d/sum;
    // A fixed pixel overlap cap is captured once at birth, not rescaled as a dab
    // grows. Large inflated supports therefore retain a nonvanishing shrink force.
    const cap=Math.max(0,settings.capPx??20),normalization=Math.max(.1,settings.normalization??20);
    const gap=d-sum,e=(gap-clamp(gap,-cap,0))/normalization,factor=e/Math.sqrt(1+e*e)/normalization;
    const gradSumN=ra.direction.map((x,k)=>x-rb.direction[k]);
    const dot=gradSumN[0]*n[0]+gradSumN[1]*n[1];
    const gd=n.map((x,k)=>factor*(x-(gradSumN[k]-dot*x)/d));
    const radialFactor=-factor;
    const shape=(radius,sign)=>({center:gd.map((x,k)=>sign*x*D[k]),
      logScale:radius.logScale.map(x=>x*radialFactor),theta:radius.theta*radialFactor});
    return {energy:Math.sqrt(1+e*e)-1,ratio,gap,capPx:cap,radii:[ra.radius,rb.radius],a:shape(ra,-1),b:shape(rb,1)};
  }

  function rowFromParams(p,i) {
    return {x:p.xy[i*2],y:p.xy[i*2+1],sx:p.scale[i*2],sy:p.scale[i*2+1],theta:p.theta[i],
      tag:p.detailTags?.[i]||0,taper:p.brushTaper?.[i]??1};
  }

  function settingsFromParams(p) {
    return {feather:.18,widthStart:p.brushWidthTaperStart??1,widthEnd:p.brushWidthTaperEnd??1,widthTaperEnabled:Boolean(p.brushWidthTaperEnabled)};
  }

  function supportShader(fixed=false,fixedScale=8192) {
    return `
struct Position {center:vec2<f32>,taper:f32,pad:f32};
struct Options {geometry:vec4<f32>,runtime:vec4<f32>,brush:vec4<f32>,shape:vec4<f32>};
struct QSample {q:f32,g:vec2<f32>};
struct Radius {r:f32,logs:vec2<f32>,theta:f32,direction:vec2<f32>,valid:f32};
@group(0) @binding(0) var<uniform> options:Options;
@group(0) @binding(1) var<storage,read> positions:array<Position>;
@group(0) @binding(2) var<storage,read> transforms:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read_write> gradients:array<atomic<${fixed?'i32':'u32'}>>;
@group(0) @binding(4) var<storage,read> neighbors:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read> colors:array<vec4<f32>>;
${fixed?'@group(0) @binding(6) var<storage,read_write> control:array<atomic<u32>>;':''}
fn load(index:u32)->f32{return ${fixed?`f32(atomicLoad(&gradients[index]))/${fixedScale}.0`:'bitcast<f32>(atomicLoad(&gradients[index]))'};}
fn add(index:u32,value:f32){
${fixed?`let raw=i32(round(clamp(value*${fixedScale}.0,-2147483000.0,2147483000.0)));
atomicMax(&control[1],u32(abs(raw)));if(abs(raw)>536870911){atomicAdd(&control[0],1u);}atomicAdd(&gradients[index],clamp(raw,-536870911,536870911));`:`var bits=atomicLoad(&gradients[index]);loop{let r=atomicCompareExchangeWeak(&gradients[index],bits,bitcast<u32>(bitcast<f32>(bits)+value));if(r.exchanged){break;}bits=r.old_value;}`}
}
fn brush_q(p:vec2<f32>,row:u32)->QSample{
  let t=transforms[row];let major=t.x>=t.y;let anisotropy=max(t.x,t.y)/max(.0001,min(t.x,t.y));
  let accent=floor(t.w)>=2.0;let ribbon=anisotropy>=2.15&&!accent;
  let len=select(select(.90,1.0,ribbon),.74,accent);
  let wb=select(select(.94,.82,ribbon),.68,accent);
  let shoulder=select(select(.10,.17,ribbon),.26,accent);
  let bias=select(select(.02,-.025,ribbon),-.08,accent);
  let bendAmount=select(select(.025,.055,ribbon),.035,accent);
  let long=select(p.y,p.x,major);let transverse=select(p.x,p.y,major);
  let u=long/len;let u2=u*u;let bend=bendAmount*u*(1.0-u2);let db=bendAmount*(1.0-3.0*u2);
  let base=wb*(1.0-shoulder*u2+bias*u);let dbase=wb*(-2.0*shoulder*u+bias);
  let taper=select(0.0,clamp(positions[row].taper,0.0,1.0),options.brush.z>.5);
  let progress=clamp(.5+.5*long,0.0,1.0);let dp=select(0.0,.5*len,long> -1.0&&long<1.0);
  let tf=options.brush.x+taper*(options.brush.y-options.brush.x)*progress;
  let raw=base*tf;let w=max(.0001,raw);
  let dw=select(0.0,dbase*tf+base*taper*(options.brush.y-options.brush.x)*dp,raw>.0001);
  let v=(transverse-bend)/w;let v2=v*v;let u3=u*u2;let u4=u2*u2;let u5=u3*u2;
  let gl=(.55*4.0*u3+.45*6.0*u5+4.0*v*v2*(-db/w-v*dw/w))/len;let gt=4.0*v*v2/w;
  return QSample(.55*u4+.45*u4*u2+v2*v2,select(vec2<f32>(gt,gl),vec2<f32>(gl,gt),major));
}
fn radius(row:u32,n:vec2<f32>)->Radius{
  let t=transforms[row];let c=cos(t.z);let s=sin(t.z);let D=options.geometry.xy;
  let ext=max(vec2<f32>(.0001),t.xy*${EXTENT});
  let v=n/D;let axis=vec2<f32>(c*v.x+s*v.y,-s*v.x+c*v.y)/ext;
  var lo=0.0;var hi=2.0*max(length(D*vec2<f32>(c,s))*ext.x,length(D*vec2<f32>(s,c))*ext.y);
  for(var k=0u;k<24u;k++){let mid=(lo+hi)*.5;if(brush_q(mid*axis,row).q>options.brush.w){hi=mid;}else{lo=mid;}}
  let r=(lo+hi)*.5;let p=r*axis;let q=brush_q(p,row);let slope=dot(q.g,axis);
  if(!(slope>1e-9&&abs(q.q-options.brush.w)<.001)){
    return Radius(0.0,vec2<f32>(0.0),0.0,vec2<f32>(0.0),0.0);
  }
  let logs=select(vec2<f32>(0.0),q.g*p/slope,t.xy*${EXTENT}>vec2<f32>(.0001));
  let theta=-(q.g.x*p.y*ext.y/ext.x-q.g.y*p.x*ext.x)/slope;
  let dn=-r*vec2<f32>(q.g.x*c/ext.x-q.g.y*s/ext.y,q.g.x*s/ext.x+q.g.y*c/ext.y)/D/slope;
  return Radius(r,logs,theta,dn,1.0);
}
fn safe_branch(row:u32)->bool{
  let t=transforms[row];let a=max(t.x,t.y)/max(.0001,min(t.x,t.y));
  return abs(t.x-t.y)>.01*max(t.x,t.y)&&(floor(t.w)>=2.0||abs(a-2.15)>.025);
}
fn layer(row:u32)->f32{return floor(clamp(fract(transforms[row].w)/.24,0.0,.999999)*options.geometry.z);}
@compute @workgroup_size(64)
fn birth_links(@builtin(global_invocation_id) id:vec3<u32>){
  let row=id.x;if(row>=u32(options.runtime.x)||!safe_branch(row)){return;}
  var center=vec2<f32>(0.0);var logs=vec2<f32>(0.0);var theta=0.0;
  var tangentGradient=0.0;var tangentCount=0.0;var signedTargetWidth=0.0;
  var neighborCenter0=vec2<f32>(0.0);var neighborCenter1=vec2<f32>(0.0);var neighborCount=0u;
  var pigmentGradient=vec3<f32>(0.0);var pigmentCount=0.0;
  for(var k=0u;k<2u;k++){
    let link=neighbors[row*2u+k];let other=i32(link.x);if(other<0||u32(other)>=u32(options.runtime.x)){continue;}
    if(layer(row)!=layer(u32(other))||!safe_branch(u32(other))){continue;}
    let q=(positions[u32(other)].center-positions[row].center)*options.geometry.xy;
    let distance=length(q);if(distance<1e-5){continue;}let n=q/distance;
    let ndcDirection=vec2<f32>(n.x/options.geometry.x,n.y/options.geometry.y);
    let targetAngle=atan2(ndcDirection.y,ndcDirection.x);
    let major=transforms[row].x>=transforms[row].y;
    let axisAngle=transforms[row].z+select(1.57079632679,0.0,major);
    tangentGradient+=0.5*sin(2.0*(axisAngle-targetAngle));tangentCount+=1.0;
    pigmentGradient+=colors[row].rgb-colors[u32(other)].rgb;pigmentCount+=1.0;
    if(signedTargetWidth==0.0&&link.w!=0.0){signedTargetWidth=link.w;}
    if(neighborCount==0u){neighborCenter0=positions[u32(other)].center;}
    else if(neighborCount==1u){neighborCenter1=positions[u32(other)].center;}
    neighborCount+=1u;
    let a=radius(row,n);let b=radius(u32(other),-n);if(a.valid<.5||b.valid<.5){continue;}
    let gap=distance-a.r-b.r;let cap=max(0.0,link.y);let normalization=max(.1,link.z);
    let e=(gap-clamp(gap,-cap,0.0))/normalization;let f=e/sqrt(1.0+e*e)/normalization;
    let gs=a.direction-b.direction;let gd=f*(n-(gs-n*dot(gs,n))/distance);
    center-=gd*options.geometry.xy;logs-=a.logs*f;theta-=a.theta*f;
  }
  if(tangentCount>0.0){theta+=options.runtime.z*tangentGradient/tangentCount;}
  let targetWidth=abs(signedTargetWidth);
  if(targetWidth>0.0){
    let t=transforms[row];let minorIsX=t.x<=t.y;
    let widthGradient=options.runtime.w*clamp(log(max(.0001,min(t.x,t.y))/targetWidth),-.7,.7);
    logs+=select(vec2<f32>(0.0,widthGradient),vec2<f32>(widthGradient,0.0),minorIsX);
  }
  if(neighborCount>=2u&&targetWidth>0.0){
    let chord=(neighborCenter1-neighborCenter0)*options.geometry.xy;
    let chordLength=length(chord);
    if(chordLength>1.0){
      let tangent=chord/chordLength;let normal=vec2<f32>(-tangent.y,tangent.x);
      let midpoint=0.5*(neighborCenter0+neighborCenter1)*options.geometry.xy;
      let centerPx=positions[row].center*options.geometry.xy;
      let signedDistance=dot(centerPx-midpoint,normal);
      let direction=select(-1.0,1.0,signedTargetWidth>0.0);
      let coherence=clamp(options.shape.x,0.0,1.0);
      let desiredFraction=.18*(1.0+coherence);
      let desiredCap=18.0+20.0*coherence;
      let desiredMinimum=1.2+1.6*coherence;
      let desired=direction*min(desiredCap,max(desiredMinimum,desiredFraction*chordLength));
      let curvatureGradient=(signedDistance-desired)/max(1.0,chordLength*chordLength);
      center+=(12.0+8.0*coherence)*curvatureGradient*normal*options.geometry.xy;
    }
  }
  let multiplier=options.runtime.y*max(.01,load(row*16u+9u));
  add(row*16u,center.x*multiplier);add(row*16u+1u,center.y*multiplier);
  add(row*16u+2u,logs.x*multiplier);add(row*16u+3u,logs.y*multiplier);add(row*16u+8u,theta*multiplier);
  if(pigmentCount>0.0){
    let pigment=options.geometry.w*pigmentGradient/pigmentCount;
    add(row*16u+4u,pigment.x*multiplier);add(row*16u+5u,pigment.y*multiplier);add(row*16u+6u,pigment.z*multiplier);
  }
}
`;
  }

  global.Image2SplatPaintBrushSupport = Object.freeze({
    EXTENT, familyOf, brushQ, brushRadius, supportPenalty,
    rowFromParams, settingsFromParams, supportShader,
  });
})(globalThis);
