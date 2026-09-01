(function installFlowBirthGraph(global) {
  // CPU sidecar only. Persistent edges use stable identities, never reusable GPU rows.
  class BirthGraph {
    constructor(count=0,{maxMembers=9,minMembers=3}={}) {
      this.nextId=1;this.rows=[];this.nodes=new Map();this.maxMembers=maxMembers;this.minMembers=minMembers;
      this.births=[];this.version=0;this.caps=new Map();this.ensureCount(count);
    }
    newNode(){const id=this.nextId++;this.nodes.set(id,new Set());return id;}
    ensureCount(count){
      if(count<this.rows.length)throw new Error('Shrinking requires explicit compaction mapping');
      while(this.rows.length<count)this.rows.push(this.newNode());
    }
    group(id){
      const visited=new Set(),stack=[id];
      while(stack.length){const next=stack.pop();if(visited.has(next)||!this.nodes.has(next))continue;
        visited.add(next);stack.push(...this.nodes.get(next));}
      return [...visited];
    }
    detachId(id){
      for(const other of this.nodes.get(id)||[]){this.nodes.get(other)?.delete(id);this.caps.delete([id,other].sort((a,b)=>a-b).join(':'));}
      this.nodes.get(id)?.clear();this.version++;
    }
    invalidateRows(rows,{reuse=true}={}){
      for(const row of new Set(rows)){
        if(!Number.isInteger(row)||row<0||row>=this.rows.length)throw new Error('Invalid reused row');
        const id=this.rows[row];this.detachId(id);
        if(reuse){this.nodes.delete(id);this.rows[row]=this.newNode();}
      }
    }
    grow(oldCount,words,step=0){
      if(oldCount!==this.rows.length)throw new Error('Birth count mismatch');
      // Reject malformed GPU lineage before allocating IDs or changing any edge.
      for(const value of words){
        const word=value>>>0,mode=word>>>30,parentRow=(word&0x3fffffff)-1;
        if((mode===1||mode===2)&&(parentRow<0||parentRow>=oldCount))throw new Error('GPU grow source outside old active rows');
      }
      this.ensureCount(oldCount+words.length);
      const usedParents=new Set(),events=[];
      for(let local=0;local<words.length;local++){
        const word=words[local]>>>0,mode=word>>>30,parentRow=(word&0x3fffffff)-1,childRow=oldCount+local;
        if(mode!==1&&mode!==2)continue; // Mode zero is residual reseeding, not ancestry.
        const parent=this.rows[parentRow],child=this.rows[childRow];
        const event={step,kind:mode===1?'split':'clone',parent,child,parentRow,childRow,linked:false};
        if(!usedParents.has(parent)&&this.nodes.get(parent).size<2&&this.group(parent).length<this.maxMembers){
          this.nodes.get(parent).add(child);this.nodes.get(child).add(parent);event.linked=true;usedParents.add(parent);
        }
        this.births.push(event);events.push(event);
      }
      this.version++;return events;
    }
    relocate(words,roles=[]){
      if(words.length!==this.rows.length)throw new Error('Relocation count mismatch');
      const destinations=[],touched=new Set();
      for(let row=0;row<words.length;row++){
        const word=words[row]>>>0;if(!word)continue;
        destinations.push(row);
        const mode=word>>>30,source=(word&0x3fffffff)-1;
        if(mode===3&&source>=0&&source<roles.length&&((roles[source]&0xe0000000)>>>0)===0x40000000)touched.add(source);
      }
      this.invalidateRows(destinations);
      this.invalidateRows([...touched].filter(i=>!destinations.includes(i)),{reuse:false});
      return {destinations,touched:[...touched]};
    }
    compact(keep,{compacted=true}={}){
      if(!compacted)return;
      const seen=new Set();
      for(const i of keep){if(!Number.isInteger(i)||i<0||i>=this.rows.length||seen.has(i))throw new Error('Invalid compaction mapping');seen.add(i);}
      const next=Array.from(keep,i=>this.rows[i]),alive=new Set(next);
      for(const id of this.rows)if(!alive.has(id)){this.detachId(id);this.nodes.delete(id);}
      this.rows=next;this.version++;
    }
    pack(accept=()=>true,{includeDormant=false}={}){
      const rowOf=new Map(this.rows.map((id,row)=>[id,row]));
      const neighbors=new Int32Array(this.rows.length*2).fill(-1),edges=[],groups=[],visited=new Set();
      for(const id of this.rows){
        if(visited.has(id))continue;const group=this.group(id);group.forEach(n=>visited.add(n));
        if(group.length<(includeDormant?2:this.minMembers))continue;
        groups.push(group);
        for(const a of group)for(const b of this.nodes.get(a)){
          if(a>=b)continue;const ra=rowOf.get(a),rb=rowOf.get(b);
          if(ra===undefined||rb===undefined)throw new Error('Stale stable ID');
          if(!accept(ra,rb))continue;
          const sa=neighbors[ra*2]===-1?0:1,sb=neighbors[rb*2]===-1?0:1;
          if(neighbors[ra*2+sa]!==-1||neighbors[rb*2+sb]!==-1)throw new Error('Branch exceeds two neighbors');
          neighbors[ra*2+sa]=rb;neighbors[rb*2+sb]=ra;edges.push({a:ra,b:rb,nodeA:a,nodeB:b});
        }
      }
      return {neighbors,edges,groups,linkedCount:new Set(edges.flatMap(e=>[e.a,e.b])).size,count:this.rows.length};
    }
  }

  global.Image2SplatPaintFlowBirthGraph = Object.freeze({ BirthGraph });
})(globalThis);
