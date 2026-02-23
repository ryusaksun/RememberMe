"""像素房间组件 — Floor796 风格侧视截面公寓 + 女性角色实时行为动画。"""

from __future__ import annotations

from nicegui import ui


class PixelRoom:
    """Floor796 风格侧视截面公寓，角色行为与聊天状态联动。"""

    def __init__(self, persona_name: str):
        self.persona_name = persona_name
        self._status_label: ui.label | None = None

    def create(self, container):
        """在给定容器中创建像素房间 Canvas。"""
        with container:
            ui.element("canvas").props(
                'id="pixel-room-canvas" width="720" height="720"'
            ).style(
                "width: 100%; max-width: 600px; aspect-ratio: 1; display: block; "
                "image-rendering: pixelated; image-rendering: crisp-edges; "
                "border: 2px solid rgba(0,255,213,0.3); "
                "box-shadow: 0 0 20px rgba(0,255,213,0.12), inset 0 0 40px rgba(0,0,0,0.3); "
                "border-radius: 4px;"
            )
            self._status_label = ui.label("[ idle... ]").classes(
                "pixel-room-status"
            ).style(
                "color: #4a5568; font-size: 11px; margin-top: 8px; "
                "text-align: center; width: 100%; letter-spacing: 1px;"
            )
        ui.add_body_html(f"<script>{_JS_ENGINE}</script>")

    async def update_state(self, state: str):
        await ui.run_javascript(
            f'if(window.pixelRoomSetState) window.pixelRoomSetState("{state}")'
        )


# ---------------------------------------------------------------------------
# JS 动画引擎（内嵌）— Floor796 风格侧视截面公寓 v2
# ---------------------------------------------------------------------------

_JS_ENGINE = r"""
(function() {
if (window._pixelRoomInit) return;
window._pixelRoomInit = true;

function _boot() {
var canvas = document.getElementById('pixel-room-canvas');
if (!canvas) { setTimeout(_boot, 200); return; }
var ctx = canvas.getContext('2d');
var W = 720, H = 720;
canvas.width = W; canvas.height = H;

/* ═══ OFFSCREEN CANVAS for static background ═══ */
var bgC = document.createElement('canvas');
bgC.width = W; bgC.height = H;
var bg = bgC.getContext('2d');
var bgOK = false;

/* ═══ COLOR UTILITIES ═══ */
function shade(hex, a) {
  var n = parseInt(hex.slice(1), 16);
  var r = Math.max(0, Math.min(255, ((n>>16)&0xff)+a));
  var g = Math.max(0, Math.min(255, ((n>>8)&0xff)+a));
  var b = Math.max(0, Math.min(255, (n&0xff)+a));
  return '#'+((1<<24)|(r<<16)|(g<<8)|b).toString(16).slice(1);
}

/* Floor796 3D box: drop-shadow → 1px outline → base → 1px highlight/dark */
function box(c, x, y, w, h, base) {
  if (w<=0||h<=0) return;
  c.fillStyle='rgba(0,0,0,0.12)'; c.fillRect(x+2,y+2,w,h);
  c.fillStyle=shade(base,-40); c.fillRect(x,y,w,h);
  c.fillStyle=base; c.fillRect(x+1,y+1,w-2,h-2);
  if(h>3&&w>3){
    c.fillStyle=shade(base,25); c.fillRect(x+1,y+1,w-2,1); c.fillRect(x+w-2,y+1,1,h-2);
    c.fillStyle=shade(base,-20); c.fillRect(x+1,y+h-2,w-2,1); c.fillRect(x+1,y+1,1,h-2);
  }
}
function fbox(c, x, y, w, h, base) {
  if(w<=0||h<=0) return;
  c.fillStyle=shade(base,-30); c.fillRect(x,y,w,h);
  c.fillStyle=base; c.fillRect(x+1,y+1,Math.max(0,w-2),Math.max(0,h-2));
  if(h>3&&w>3){
    c.fillStyle=shade(base,20); c.fillRect(x+1,y+1,w-2,1); c.fillRect(x+w-2,y+1,1,h-2);
    c.fillStyle=shade(base,-15); c.fillRect(x+1,y+h-2,w-2,1); c.fillRect(x+1,y+1,1,h-2);
  }
}
function fr(c,x,y,w,h,col){c.fillStyle=col;c.fillRect(x,y,w,h);}

/* ═══ LAYOUT ═══
   Thicker structure for Floor796 feel:
   Roof=20, Upper=316, Mid=48(pipes!), Lower=316, Ground=20
   Center wall=24px wide
*/
var ROOF=20, UH=316, MID=48, LH=316, GND=20;
var UY=ROOF, MY=ROOF+UH, LY=MY+MID, GY=LY+LH;
var CX=348, CW=24, WP=8;
var UF=MY-2, LF=GY-2;
var DOOR_H=70;
var UDOOR_Y=MY-DOOR_H, LDOOR_Y=GY-DOOR_H;
var STX=260, STW=80;

/* ═══ FLOOR/TILE HELPERS ═══ */
function drawWoodFloor(c,x,y,w,h,base){
  fr(c,x,y,w,h,shade(base,-8));
  for(var px=x;px<x+w;px+=32){
    var tw=Math.min(32,x+w-px);
    fr(c,px,y,tw,h,base);
    fr(c,px,y,1,h,shade(base,-18));
    fr(c,px,y,tw,1,shade(base,12));
    /* Wood grain */
    for(var g=0;g<3;g++){
      var gx=px+4+(g*11)%24, gy=y+1+(g*7)%Math.max(1,h-2);
      if(gx<px+tw-4) fr(c,gx,gy,6+(g*5)%10,1,shade(base,-6));
    }
  }
  fr(c,x,y+h-1,w,1,shade(base,-25));
}
function drawTileFloor(c,x,y,w,h,base){
  var ts=14;
  for(var ty=y;ty<y+h;ty+=ts){
    for(var tx=x;tx<x+w;tx+=ts){
      var tw=Math.min(ts,x+w-tx),th=Math.min(ts,y+h-ty);
      fr(c,tx,ty,tw,th,base);
      fr(c,tx,ty,tw,1,shade(base,-10));
      fr(c,tx,ty,1,th,shade(base,-10));
      fr(c,tx+tw-1,ty,1,th,shade(base,8));
      fr(c,tx,ty+th-1,tw,1,shade(base,8));
    }
  }
}

/* ═══ WALL TEXTURE — adds panel lines, baseboard, crown ═══ */
function drawWallTexture(c,x,y,w,h,base,wainBase){
  fr(c,x,y,w,h,base);
  /* Horizontal panel grooves */
  for(var py=y+40;py<y+h-100;py+=50){
    fr(c,x,py,w,1,shade(base,-10));
    fr(c,x,py+1,w,1,shade(base,8));
  }
  /* Wainscoting lower 90px */
  var wy=y+h-90;
  fr(c,x,wy,w,90,wainBase);
  fr(c,x,wy,w,2,shade(wainBase,-15));
  /* Wainscot panel pattern */
  for(var px=x+20;px<x+w-20;px+=60){
    var pw=Math.min(50,x+w-px-10);
    if(pw>10){
      fr(c,px,wy+8,pw,72,shade(wainBase,8));
      fr(c,px,wy+8,pw,1,shade(wainBase,18));
      fr(c,px+pw-1,wy+8,1,72,shade(wainBase,18));
      fr(c,px,wy+79,pw,1,shade(wainBase,-10));
      fr(c,px,wy+8,1,72,shade(wainBase,-10));
    }
  }
  /* Crown molding at top */
  fr(c,x,y,w,3,shade(base,15));
  fr(c,x,y+3,w,1,shade(base,-10));
  /* Baseboard at bottom */
  fr(c,x,y+h-6,w,6,shade(wainBase,-20));
  fr(c,x,y+h-6,w,1,shade(wainBase,-10));
}

/* ═══ CEILING STRIP with light fixture ═══ */
function drawCeiling(c,x,y,w,lightX,lightW){
  /* Ceiling structure */
  fr(c,x,y,w,8,shade('#a09888',-5));
  fr(c,x,y,w,1,shade('#a09888',15));
  fr(c,x,y+7,w,1,shade('#a09888',-15));
  /* Ceiling beams */
  for(var bx=x+40;bx<x+w;bx+=80){
    fbox(c,bx,y,6,8,'#948878');
  }
  /* Light fixture */
  if(lightW>0){
    fbox(c,lightX,y+4,lightW,5,'#d8d0c0');
    fr(c,lightX+2,y+7,lightW-4,2,'#f0e8d0');
    /* Light glow */
    c.fillStyle='rgba(255,245,220,0.04)';
    c.fillRect(lightX-20,y+8,lightW+40,100);
  }
}

/* ═══ STRUCTURE ═══ */
function drawCenterWall(c,x,y,w,h,doorY,doorH){
  fr(c,x,y,w,h,shade('#606870',-20));
  fr(c,x+1,y,w-2,h,'#606870');
  fr(c,x+w-2,y,1,h,shade('#606870',15));
  fr(c,x+1,y,1,h,shade('#606870',-10));
  /* Vertical panel lines */
  fr(c,x+8,y,1,h,shade('#606870',-8));
  fr(c,x+16,y,1,h,shade('#606870',-8));
  /* Door opening */
  fr(c,x,doorY,w,doorH,'#2a2428');
  /* Door frame — thick wood */
  fbox(c,x-3,doorY-6,w+6,7,'#8a7a68');
  fbox(c,x-3,doorY,4,doorH,'#8a7a68');
  fbox(c,x+w-1,doorY,4,doorH,'#8a7a68');
  /* Door arch detail */
  fr(c,x+1,doorY-1,w-2,1,'#7a6a58');
}

function drawMidFloorPipes(c){
  var y0=MY, h=MID;
  /* Top and bottom surfaces */
  fr(c,0,y0,W,h,'#505060');
  fr(c,0,y0,W,4,'#606878');
  fr(c,0,y0,W,1,shade('#606878',15));
  fr(c,0,y0+h-4,W,4,'#505868');
  fr(c,0,y0+h-1,W,1,shade('#505868',-15));
  /* Background darker */
  fr(c,0,y0+4,W,h-8,'#404050');
  /* Pipes! */
  /* Green water pipe */
  fr(c,0,y0+8,W,8,'#507050');
  fr(c,0,y0+8,W,1,'#608060');
  fr(c,0,y0+15,W,1,'#406040');
  /* Joints on green pipe */
  for(var jx=60;jx<W;jx+=120){fbox(c,jx,y0+6,10,12,'#608868');}
  /* Red hot water pipe */
  fr(c,0,y0+20,W,6,'#806050');
  fr(c,0,y0+20,W,1,'#907060');
  fr(c,0,y0+25,W,1,'#604840');
  for(var jx=100;jx<W;jx+=140){fbox(c,jx,y0+18,8,10,'#a07868');}
  /* Gray ventilation duct */
  fr(c,0,y0+30,W,10,'#686870');
  fr(c,0,y0+30,W,1,'#787880');
  fr(c,0,y0+39,W,1,'#585860');
  /* Duct rivets */
  for(var rx=20;rx<W;rx+=30){fr(c,rx,y0+34,2,2,'#787880');}
  /* Yellow electrical conduit */
  fr(c,0,y0+43,W,3,'#a09048');
  fr(c,0,y0+43,W,1,'#b0a058');
  /* Support brackets */
  for(var bx=50;bx<W;bx+=90){
    fbox(c,bx,y0+4,4,h-8,'#585868');
  }
  /* Staircase opening */
  fr(c,STX,y0,STW,h,'#1a1618');
  /* Stair wall back */
  fr(c,STX+1,y0+2,STW-2,h-4,'#282428');
  /* Stair steps — thicker with more detail */
  var ns=8;
  for(var i=0;i<ns;i++){
    var sx=STX+i*(STW/ns);
    var sy=y0+i*(h/ns);
    var sw=Math.ceil(STW/ns)+1;
    var sh=h-Math.floor(i*h/ns);
    /* Step riser (vertical face) */
    fr(c,sx,sy,sw,sh,'#686060');
    /* Step tread (top surface) - lighter */
    fbox(c,sx,sy,sw,5,'#908880');
    fr(c,sx,sy,sw,1,'#a09888');
    /* Anti-slip strip */
    fr(c,sx+2,sy+1,sw-4,1,'#e0d8c0');
  }
  /* Stair stringers (side beams) */
  fr(c,STX,y0,2,h,'#504848');
  fr(c,STX+STW-2,y0,2,h,'#504848');
  /* Under-stair storage door (visible from bedroom side) */
  fbox(c,STX+5,y0+h-28,24,26,'#706858');
  fr(c,STX+25,y0+h-18,4,6,'#a09080');
}

function drawStructure(c){
  fr(c,0,0,W,H,'#18161a');

  /* Room backgrounds with wall texture */
  drawWallTexture(c,WP,UY,CX-WP,UH,'#c8b8a8','#b8a898');
  drawWallTexture(c,CX+CW,UY,W-WP-CX-CW,UH,'#b8c8c8','#a8b8b8');
  drawWallTexture(c,WP,LY,CX-WP,LH,'#c0b0a0','#b0a090');
  drawWallTexture(c,CX+CW,LY,W-WP-CX-CW,LH,'#c8c0a8','#b8b098');

  /* Bathroom wall tiles (upper portion) */
  for(var ty=UY+8;ty<MY-100;ty+=14){
    for(var tx=CX+CW+4;tx<W-WP-4;tx+=18){
      var off=(Math.floor(ty/14)%2)*9;
      var tw=Math.min(16,W-WP-4-tx-off);
      if(tw>4){
        fr(c,tx+off,ty,tw,12,'#a8c0b8');
        fr(c,tx+off,ty,tw,1,'#98b0a8');
        fr(c,tx+off,ty,1,12,'#98b0a8');
        fr(c,tx+off+tw-1,ty,1,12,shade('#a8c0b8',10));
        fr(c,tx+off,ty+11,tw,1,shade('#a8c0b8',10));
      }
    }
  }

  /* Kitchen backsplash */
  for(var ty=GY-100;ty<GY-50;ty+=10){
    for(var tx=CX+CW+4;tx<W-WP-4;tx+=14){
      var off=(Math.floor(ty/10)%2)*7;
      var tw=Math.min(12,W-WP-4-tx-off);
      if(tw>4){
        fr(c,tx+off,ty,tw,8,'#c0b8a0');
        fr(c,tx+off,ty,tw,1,'#b0a890');
        fr(c,tx+off,ty,1,8,'#b0a890');
      }
    }
  }

  /* Ceiling strips with lights */
  drawCeiling(c,WP,UY,CX-WP,WP+120,60);
  drawCeiling(c,CX+CW,UY,W-WP-CX-CW,CX+CW+140,50);
  drawCeiling(c,WP,LY,CX-WP,WP+100,60);
  drawCeiling(c,CX+CW,LY,W-WP-CX-CW,CX+CW+120,50);

  /* Floors */
  drawWoodFloor(c,WP,UF-4,CX-WP,MY-UF+4,'#b89870');
  drawTileFloor(c,CX+CW,UF-4,W-WP-CX-CW,MY-UF+4,'#98b8ae');
  drawWoodFloor(c,WP,LF-4,CX-WP,GY-LF+4,'#b89870');
  drawTileFloor(c,CX+CW,LF-4,W-WP-CX-CW,GY-LF+4,'#b0a888');

  /* Roof — thick with industrial detail */
  box(c,0,0,W,ROOF,'#585060');
  for(var i=0;i<W;i+=20){fr(c,i,4,18,4,shade('#585060',-8));}
  for(var i=10;i<W;i+=20){fr(c,i,12,18,3,shade('#585060',5));}
  /* Roof rivets */
  for(var rx=15;rx<W;rx+=40){fr(c,rx,8,2,2,'#686068');}

  /* Ground */
  box(c,0,GY,W,GND,'#585060');
  for(var i=0;i<W;i+=24){fr(c,i,GY+4,22,4,shade('#585060',-6));}

  /* Mid-floor with pipes */
  drawMidFloorPipes(c);

  /* Stair railings — upper room (industrial metal) */
  fbox(c,STX-4,MY-44,4,46,'#686060');
  fbox(c,STX+STW,MY-44,4,46,'#686060');
  fbox(c,STX-4,MY-44,STW+8,4,'#787070');
  /* horizontal mid-bars */
  fr(c,STX-4,MY-30,STW+8,2,'#787070');
  fr(c,STX-4,MY-18,STW+8,2,'#787070');
  /* vertical balusters */
  for(var bx=STX+14;bx<STX+STW-4;bx+=16){fr(c,bx,MY-42,2,42,'#686060');}
  /* Stair railings — lower room */
  fbox(c,STX-4,LY,4,44,'#686060');
  fbox(c,STX+STW,LY,4,44,'#686060');
  fbox(c,STX-4,LY+40,STW+8,4,'#787070');
  fr(c,STX-4,LY+26,STW+8,2,'#787070');
  fr(c,STX-4,LY+14,STW+8,2,'#787070');
  for(var bx=STX+14;bx<STX+STW-4;bx+=16){fr(c,bx,LY,2,42,'#686060');}

  /* Center walls */
  drawCenterWall(c,CX,UY,CW,UH,UDOOR_Y,DOOR_H);
  drawCenterWall(c,CX,LY,CW,LH,LDOOR_Y,DOOR_H);

  /* Outer walls — thick with industrial detail */
  box(c,0,UY,WP,UH+MID+LH,'#606870');
  box(c,W-WP,UY,WP,UH+MID+LH,'#606870');
  /* Wall rivets */
  for(var ry=UY+20;ry<GY;ry+=40){
    fr(c,2,ry,2,2,'#707880');
    fr(c,W-4,ry,2,2,'#707880');
  }
  /* Vertical conduit on left wall */
  fr(c,3,UY+40,3,UH-60,'#505860');
  fr(c,3,UY+40,3,1,'#607070');
  /* Junction boxes */
  fbox(c,1,UY+80,6,8,'#586068');fr(c,2,UY+82,4,4,'#68707a');
  fbox(c,1,LY+60,6,8,'#586068');fr(c,2,LY+62,4,4,'#68707a');
  /* Vertical conduit on right wall */
  fr(c,W-6,UY+60,3,UH-80,'#505860');
  fbox(c,W-7,UY+120,6,8,'#586068');
  fbox(c,W-7,LY+100,6,8,'#586068');

  /* ═══ AMBIENT WINDOW LIGHT GRADIENTS ═══ */
  /* Bedroom window light — brightens right area */
  var wlg=c.createLinearGradient(WP+90,0,WP+280,0);
  wlg.addColorStop(0,'rgba(255,248,230,0)');
  wlg.addColorStop(0.2,'rgba(255,248,230,0.06)');
  wlg.addColorStop(0.5,'rgba(255,248,230,0.03)');
  wlg.addColorStop(1,'rgba(0,0,0,0)');
  c.fillStyle=wlg;c.fillRect(WP,UY,CX-WP,UH);
  /* Window light cone on floor */
  c.fillStyle='rgba(255,248,230,0.05)';c.fillRect(WP+80,UF-20,120,22);

  /* Living room window light */
  var wlg2=c.createLinearGradient(WP,0,WP+140,0);
  wlg2.addColorStop(0,'rgba(200,220,255,0.06)');
  wlg2.addColorStop(1,'rgba(200,220,255,0)');
  c.fillStyle=wlg2;c.fillRect(WP,LY,CX-WP,LH);
  c.fillStyle='rgba(200,220,255,0.04)';c.fillRect(WP+10,LF-18,100,20);

  /* Kitchen window light (evening glow) */
  var wlg3=c.createLinearGradient(W-WP-100,0,W-WP,0);
  wlg3.addColorStop(0,'rgba(255,220,180,0)');
  wlg3.addColorStop(1,'rgba(255,220,180,0.06)');
  c.fillStyle=wlg3;c.fillRect(CX+CW,LY,W-WP-CX-CW,LH);

  /* Dark corners (ambient occlusion) */
  c.fillStyle='rgba(0,0,0,0.04)';
  c.fillRect(WP,UY,20,30);c.fillRect(CX-20,UY,20,30);
  c.fillRect(WP,MY-30,20,30);c.fillRect(CX-20,MY-30,20,30);
  c.fillRect(WP,LY,20,30);c.fillRect(CX-20,LY,20,30);
  c.fillRect(WP,GY-30,20,30);c.fillRect(CX-20,GY-30,20,30);
  c.fillRect(CX+CW,UY,20,30);c.fillRect(W-WP-20,UY,20,30);
  c.fillRect(CX+CW,LY,20,30);c.fillRect(W-WP-20,LY,20,30);
}

/* ═══ BEDROOM (upper-left) ═══ */
function drawBedroom(c){
  var rx=WP,ry=UY+8;

  /* ── Window + curtains ── */
  var wx=rx+100,wy=ry+24,ww=100,wh=80;
  fbox(c,wx-5,wy-5,ww+10,wh+10,'#706858');
  fr(c,wx,wy,ww,wh,'#8ab8d8');
  fr(c,wx,wy+wh*0.55|0,ww,wh*0.45|0,'#78a8c8');
  /* City silhouette */
  fr(c,wx,wy+wh*0.4|0,ww,wh*0.15|0,'#6090b0');
  /* Buildings */
  fr(c,wx+4,wy+wh*0.3|0,8,wh*0.28|0,'#5080a0');
  fr(c,wx+16,wy+wh*0.35|0,6,wh*0.22|0,'#5888a8');
  fr(c,wx+28,wy+wh*0.25|0,10,wh*0.32|0,'#5080a0');
  fr(c,wx+42,wy+wh*0.38|0,8,wh*0.2|0,'#5888a8');
  fr(c,wx+56,wy+wh*0.22|0,6,wh*0.35|0,'#4878a0');
  fr(c,wx+66,wy+wh*0.3|0,12,wh*0.28|0,'#5080a0');
  fr(c,wx+82,wy+wh*0.32|0,8,wh*0.26|0,'#5888a8');
  /* Building windows (tiny lit squares) */
  fr(c,wx+6,wy+wh*0.33|0,2,2,'#e0d878');fr(c,wx+30,wy+wh*0.28|0,2,2,'#e8e090');
  fr(c,wx+58,wy+wh*0.26|0,2,2,'#e0d878');fr(c,wx+68,wy+wh*0.34|0,2,2,'#f0e898');
  fr(c,wx+84,wy+wh*0.36|0,2,2,'#e0d878');
  /* Clouds */
  fr(c,wx+12,wy+10,24,8,'#b8d0e8');fr(c,wx+16,wy+6,16,6,'#b8d0e8');
  fr(c,wx+60,wy+16,20,6,'#b8d0e8');
  /* Window cross */
  fr(c,wx+ww/2-1,wy,2,wh,'#706858');
  fr(c,wx,wy+wh/2-1,ww,2,'#706858');
  /* Window sill */
  fbox(c,wx-8,wy+wh,ww+16,6,'#706858');
  /* Small plant on sill */
  fbox(c,wx+ww-22,wy+wh-8,10,8,'#907858');
  fr(c,wx+ww-20,wy+wh-14,6,8,'#68a868');
  fr(c,wx+ww-24,wy+wh-12,4,6,'#58a058');
  /* Curtains */
  fbox(c,wx-20,wy-5,17,wh+38,'#d8b8c0');
  fr(c,wx-18,wy,2,wh+28,'#c8a8b0');fr(c,wx-12,wy,2,wh+28,'#c8a8b0');
  fr(c,wx-8,wy+2,2,wh+24,'#c8a8b0');
  fbox(c,wx+ww+3,wy-5,17,wh+38,'#d8b8c0');
  fr(c,wx+ww+5,wy,2,wh+28,'#c8a8b0');fr(c,wx+ww+11,wy,2,wh+28,'#c8a8b0');
  fr(c,wx+ww+15,wy+2,2,wh+24,'#c8a8b0');
  /* Curtain ties */
  fbox(c,wx-18,wy+wh+20,12,6,'#c0a0a8');
  fbox(c,wx+ww+5,wy+wh+20,12,6,'#c0a0a8');
  /* Curtain rod */
  fr(c,wx-22,wy-7,ww+44,2,'#807068');

  /* ── Wall art / posters ── */
  fbox(c,rx+18,ry+30,36,46,'#e0d0c0');
  fr(c,rx+21,ry+33,30,40,'#d0c0b0');
  fr(c,rx+21,ry+33,30,22,'#90c0d8');
  fr(c,rx+21,ry+49,30,10,'#70a088');
  fr(c,rx+28,ry+40,10,19,'#80b080');
  /* Second poster */
  fbox(c,rx+62,ry+38,28,36,'#d0c0b0');
  fr(c,rx+65,ry+41,22,30,'#c0b0a0');
  fr(c,rx+68,ry+44,16,8,'#d88080');
  fr(c,rx+71,ry+56,10,12,'#e8c8a0');

  /* ── Wall clock ── */
  fbox(c,rx+260,ry+30,20,20,'#e0d8d0');
  fr(c,rx+262,ry+32,16,16,'#f0e8e0');
  fr(c,rx+269,ry+34,2,8,'#404040');
  fr(c,rx+269,ry+40,6,2,'#404040');
  fr(c,rx+270,ry+40,1,1,'#c04040');

  /* ── Wall shelf ── */
  fbox(c,rx+290,ry+70,40,4,'#8a7a68');
  fbox(c,rx+290,ry+74,3,10,'#7a6a58');
  fbox(c,rx+327,ry+74,3,10,'#7a6a58');
  /* Items on shelf */
  fbox(c,rx+293,ry+62,8,8,'#c88080');
  fbox(c,rx+303,ry+58,6,12,'#80a0c8');
  fbox(c,rx+311,ry+64,10,6,'#e8d8b0');
  fbox(c,rx+323,ry+60,5,10,'#a088c0');

  /* ── Bookshelf (tall, on right side) ── */
  var bsx=rx+290,bsy=ry+90;
  box(c,bsx,bsy,42,120,'#806848');
  var bkC=['#c05050','#5080b0','#50a050','#c0a040','#8868a8','#c07040'];
  for(var si=0;si<4;si++){
    var sy=bsy+4+si*29;
    fr(c,bsx+3,sy+24,36,3,'#806848');
    var bk=bsx+5;
    for(var bi=0;bi<4;bi++){
      var bw=5+(bi*3)%4;
      fbox(c,bk,sy,bw,23,bkC[(bi+si*2)%6]);
      bk+=bw+1;
    }
  }

  /* ── Desk (larger) ── */
  var dx=rx+20,dy=UF-58;
  box(c,dx,dy,100,6,'#806848');
  box(c,dx+4,dy+6,6,52,'#705838');
  box(c,dx+90,dy+6,6,52,'#705838');
  /* Back panel */
  fr(c,dx+10,dy+6,80,4,'#705838');
  /* Drawers right */
  fbox(c,dx+60,dy+12,32,20,'#907858');
  fr(c,dx+72,dy+19,8,3,'#b09878');
  fbox(c,dx+60,dy+34,32,20,'#907858');
  fr(c,dx+72,dy+41,8,3,'#b09878');

  /* ── Monitor on desk ── */
  var mx=dx+18,my=dy-48;
  box(c,mx,my,56,42,'#282830');
  fr(c,mx+3,my+3,50,34,'#1a3040');
  fbox(c,mx+24,my+42,8,6,'#383840');
  fbox(c,mx+18,my+46,20,4,'#383840');
  /* Screen content — code lines */
  var lc=['#70b890','#90b0d0','#d0b070','#b090c0','#90b0d0','#d09090','#70b890','#b0b0d0'];
  for(var i=0;i<8;i++){
    fr(c,mx+6,my+6+i*4,8+((i*17)%26),2,lc[i]);
  }

  /* ── Keyboard + mouse ── */
  fbox(c,dx+16,dy-5,40,5,'#383840');
  for(var k=0;k<5;k++)fr(c,dx+19+k*7,dy-4,5,2,'#484850');
  for(var k=0;k<4;k++)fr(c,dx+22+k*7,dy-2,5,2,'#484850');
  /* Mouse */
  fbox(c,dx+60,dy-4,7,5,'#d8d0c8');
  fr(c,dx+60,dy-4,7,1,'#c0b8b0');

  /* ── Cup on desk ── */
  fbox(c,dx+82,dy-8,8,8,'#e0d0c0');
  fr(c,dx+84,dy-6,4,4,'#c8a878');
  fbox(c,dx+89,dy-5,3,4,'#d0c0b0');

  /* ── Papers on desk ── */
  fr(c,dx+70,dy-3,12,4,'#e8e0d8');
  fr(c,dx+72,dy-4,10,4,'#f0e8e0');

  /* ── Chair ── */
  var cx=dx+34,cy=UF-38;
  box(c,cx,cy-28,32,28,'#c87888');
  fr(c,cx+4,cy-24,24,20,'#d88898');
  box(c,cx+2,cy-2,28,8,'#c87888');
  fbox(c,cx+4,cy+6,4,30,'#685848');
  fbox(c,cx+26,cy+6,4,30,'#685848');
  /* Chair wheels */
  fr(c,cx+2,cy+34,4,3,'#484848');
  fr(c,cx+26,cy+34,4,3,'#484848');
  fr(c,cx+14,cy+34,4,3,'#484848');

  /* ── Bed (right side, large) ── */
  var bx=rx+170,by=UF-54;
  /* Headboard */
  box(c,bx+110,by-14,14,68,'#806848');
  fr(c,bx+113,by-10,8,16,'#907858');
  /* Bed frame */
  box(c,bx,by,124,54,'#907858');
  /* Mattress */
  fr(c,bx+4,by+4,116,24,'#e8e0d4');
  fr(c,bx+4,by+4,116,2,'#d8d0c4');
  /* Blanket */
  fr(c,bx+4,by+16,108,26,'#d898a8');
  fr(c,bx+4,by+16,108,3,'#c08090');
  /* Blanket folds */
  for(var i=0;i<5;i++){fr(c,bx+8+i*20,by+22,14,10,'#e0a8b8');}
  /* Pillow */
  box(c,bx+84,by+6,28,14,'#e8e0d4');
  fr(c,bx+86,by+8,24,10,'#f0e8dc');
  /* Second pillow */
  box(c,bx+68,by+7,18,12,'#e0d8cc');
  /* Bed legs */
  fbox(c,bx+2,UF-8,8,8,'#705838');
  fbox(c,bx+114,UF-8,8,8,'#705838');

  /* ── Nightstand ── */
  var nx=bx-32,ny=UF-36;
  box(c,nx,ny,28,36,'#806848');
  fbox(c,nx+3,ny+4,22,14,'#907858');
  fr(c,nx+10,ny+9,8,3,'#b09878');
  fbox(c,nx+3,ny+20,22,12,'#907858');
  fr(c,nx+10,ny+24,8,3,'#b09878');
  /* Lamp on nightstand */
  fbox(c,nx+8,ny-24,12,24,'#e0d0a8');
  fr(c,nx+10,ny-22,8,20,'#e8d8b0');
  fbox(c,nx+12,ny-2,4,4,'#706858');
  c.fillStyle='rgba(255,240,200,0.05)';c.fillRect(nx-15,ny-34,58,55);
  /* Alarm clock */
  fbox(c,nx+20,ny-8,8,7,'#404848');
  fr(c,nx+21,ny-7,6,4,'#40c060');

  /* ── Rug ── */
  fr(c,rx+30,UF-5,140,7,'#c8a888');
  fr(c,rx+32,UF-4,136,5,'#b89878');
  fr(c,rx+34,UF-3,132,3,'#c8b090');
  /* Rug pattern */
  for(var i=0;i<6;i++){fr(c,rx+40+i*20,UF-3,8,3,shade('#c8b090',10));}

  /* ── Power strip on floor ── */
  fbox(c,dx+8,UF-5,18,4,'#e0d8d0');
  fr(c,dx+10,UF-4,3,2,'#404040');fr(c,dx+15,UF-4,3,2,'#404040');
  fr(c,dx+20,UF-4,3,2,'#404040');
  /* Cable */
  fr(c,dx+12,UF-5,1,3,'#404040');

  /* ── Wall outlet ── */
  fbox(c,rx+50,ry+160,8,10,'#d8d0c8');
  fr(c,rx+52,ry+162,2,2,'#404040');fr(c,rx+54,ry+162,2,2,'#404040');

  /* ── Wardrobe (against right wall near center) ── */
  var wrx=rx+234,wry=ry+70;
  box(c,wrx,wry,50,148,'#806848');
  fbox(c,wrx+3,wry+3,20,142,'#907858');
  fbox(c,wrx+26,wry+3,21,142,'#907858');
  fr(c,wrx+22,wry+50,4,12,'#b09878');
  fr(c,wrx+47,wry+50,0,12,'#b09878');
  fr(c,wrx+24,wry+50,1,12,'#b09878');
  /* Wardrobe top decoration */
  fbox(c,wrx+5,wry-8,16,8,'#c0b0a0');
  fbox(c,wrx+28,wry-8,12,8,'#a0c0d0');

  /* ── Extra detail: slippers on floor ── */
  fr(c,rx+140,UF-3,7,3,'#d8b0b0');fr(c,rx+150,UF-3,7,3,'#d8b0b0');
  /* Cable from desk to outlet */
  fr(c,rx+28,UF-4,1,2,'#404040');fr(c,rx+28,UF-5,22,1,'#404040');fr(c,rx+50,UF-5,1,35,'#404040');
  /* Sticky notes on monitor bezel */
  fr(c,rx+18+56+2,ry+120,6,6,'#e8e060');fr(c,rx+18+56+2,ry+128,6,5,'#60c8e8');
  /* Headphone on desk */
  fbox(c,rx+90,UF-58-8,14,5,'#303038');
  fr(c,rx+90,UF-58-12,2,6,'#404048');fr(c,rx+102,UF-58-12,2,6,'#404048');
  fr(c,rx+90,UF-58-14,14,3,'#404048');
  /* Trash can under desk */
  fbox(c,rx+96,UF-16,12,16,'#808080');fr(c,rx+98,UF-14,8,12,'#707070');
  /* Wall switch near door */
  fbox(c,rx+324,ry+120,6,8,'#d8d0c8');fr(c,rx+325,ry+122,4,2,'#a0a0a0');
  /* Calendar on wall */
  fbox(c,rx+120,ry+36,18,22,'#e8e0d8');fr(c,rx+122,ry+38,14,4,'#d06060');
  fr(c,rx+123,ry+44,4,3,'#404040');fr(c,rx+129,ry+44,4,3,'#404040');
  fr(c,rx+123,ry+49,4,3,'#707070');fr(c,rx+129,ry+49,4,3,'#707070');

  /* ── Fairy lights string above bed ── */
  fr(c,rx+170,ry+58,130,1,'#685848');
  var flCols=['#e06060','#e0c040','#60c060','#6080e0','#e060a0'];
  for(var fi=0;fi<8;fi++){fr(c,rx+176+fi*15,ry+56,3,3,flCols[fi%5]);}

  /* ── Stuffed animal on bed ── */
  fbox(c,rx+180,UF-54+8,10,10,'#e8d0a8');
  fr(c,rx+182,UF-54+6,6,4,'#e8d0a8');fr(c,rx+183,UF-54+5,2,1,'#303030');
  fr(c,rx+186,UF-54+5,2,1,'#303030');

  /* ── Backpack on floor ── */
  fbox(c,rx+8,UF-22,14,20,'#c05858');fr(c,rx+10,UF-20,10,14,'#d06868');
  fbox(c,rx+12,UF-24,6,4,'#a04848');

  /* ── Desk cable management ── */
  fr(c,rx+24,UF-4,1,2,'#383838');fr(c,rx+24,UF-2,40,1,'#383838');
  fr(c,rx+64,UF-2,1,2,'#383838');

  /* ── LED strip under monitor ── */
  fr(c,rx+36,UF-58-1,50,1,'#6040c0');
}

/* ═══ BATHROOM (upper-right) ═══ */
function drawBathroom(c){
  var rx=CX+CW, ry=UY+8;

  /* ── Sink counter + Mirror ── */
  var sx=rx+16,sy=UF-48;
  /* Mirror */
  fbox(c,sx+6,ry+50,44,56,'#909ca0');
  fr(c,sx+9,ry+53,38,50,'#b8ccd4');
  fr(c,sx+9,ry+53,38,12,'#c8dce4');
  /* Mirror frame detail */
  fr(c,sx+9,ry+53,38,1,'#a0b0b8');
  /* Counter */
  box(c,sx,sy,56,24,'#d8dce0');
  fr(c,sx+4,sy+4,48,14,'#b8c4cc');
  /* Faucet */
  fbox(c,sx+22,sy-10,4,12,'#a0aab0');
  fbox(c,sx+16,sy-12,18,4,'#a8b4b8');
  /* Water droplet */
  fr(c,sx+25,sy-2,2,3,'rgba(100,160,220,0.4)');
  /* Pedestal */
  fbox(c,sx+18,sy+24,20,24,'#c8ccd0');

  /* ── Soap dispenser ── */
  fbox(c,sx+44,sy-6,6,6,'#d8c8a0');
  fbox(c,sx+45,sy-10,4,4,'#c8b890');

  /* ── Toilet ── */
  var tx=rx+100,ty=UF-46;
  box(c,tx,ty+16,34,30,'#d8dce0');
  fr(c,tx+4,ty+20,26,22,'#c8ccd0');
  box(c,tx+2,ty,30,18,'#d8dce0');
  fr(c,tx+6,ty+4,22,10,'#ccd0d4');
  fbox(c,tx+30,ty+6,6,4,'#a0aab0');
  fbox(c,tx+2,ty+14,32,4,'#e0e4e8');
  /* TP holder */
  fbox(c,tx+36,ty+24,3,12,'#a0aab0');
  fbox(c,tx+33,ty+22,8,3,'#a0aab0');
  fr(c,tx+33,ty+24,6,8,'#e8e0d8');

  /* ── Towel rack ── */
  fbox(c,rx+82,ry+62,3,50,'#a0aab0');
  fbox(c,rx+80,ry+60,7,3,'#a0aab0');
  fr(c,rx+74,ry+64,18,28,'#e0a0a0');
  fr(c,rx+76,ry+66,14,24,'#d89090');
  fr(c,rx+74,ry+96,18,16,'#a0c0e0');
  fr(c,rx+76,ry+98,14,12,'#90b0d0');

  /* ── Bathtub (large) ── */
  var bx=rx+170,bby=UF-52;
  box(c,bx,bby,150,52,'#d8dce0');
  fr(c,bx+5,bby+5,140,38,'#b8c8d8');
  fr(c,bx+5,bby+5,140,10,'#c0d0e0');
  fbox(c,bx,bby,150,7,'#e0e4e8');
  /* Tub rim highlight */
  fr(c,bx+1,bby+1,148,1,'#e8ecf0');
  /* Shower pipe + head */
  fbox(c,bx+136,ry+30,4,bby-ry-30,'#a0aab0');
  fbox(c,bx+126,ry+28,18,6,'#a8b4b8');
  fbox(c,bx+130,ry+34,10,12,'#b0bcc0');
  /* Shower water dots */
  for(var d=0;d<3;d++){fr(c,bx+132+d*3,ry+46+d*8,2,4,'rgba(160,200,240,0.15)');}
  /* Curtain rod */
  fbox(c,bx+80,ry+26,70,3,'#a0aab0');
  for(var ri=0;ri<5;ri++){fbox(c,bx+84+ri*14,ry+24,4,4,'#909aa0');}
  /* Curtain */
  fbox(c,bx+120,ry+28,4,bby+8-ry-28,'#c8dcd4');
  for(var ci=0;ci<5;ci++){fr(c,bx+122,ry+36+ci*18,2,14,'#b8ccc4');}
  /* Rubber duck */
  fr(c,bx+40,bby-5,10,8,'#e8c838');
  fr(c,bx+48,bby-8,8,7,'#e8c838');
  fr(c,bx+54,bby-6,4,3,'#d0a020');
  fr(c,bx+49,bby-7,2,1,'#282828');

  /* ── Bath mat ── */
  fr(c,rx+145,UF-5,50,7,'#88c8b8');
  fr(c,rx+147,UF-4,46,5,'#78b8a8');

  /* ── Shelf with bottles ── */
  fbox(c,rx+148,ry+50,40,4,'#a0aab0');
  fbox(c,rx+150,ry+38,6,12,'#78b898');
  fbox(c,rx+158,ry+40,5,10,'#b890b8');
  fbox(c,rx+165,ry+42,6,8,'#d8b870');
  fbox(c,rx+173,ry+36,5,14,'#90b0d0');

  /* ── Small laundry basket ── */
  fbox(c,rx+145,UF-28,22,24,'#a89878');
  fr(c,rx+147,UF-26,18,20,'#988868');
  /* Clothes sticking out */
  fr(c,rx+149,UF-30,6,4,'#d090a0');
  fr(c,rx+157,UF-32,5,5,'#90a0c0');

  /* ── Bathroom scale ── */
  fbox(c,rx+120,UF-6,20,6,'#d0d4d8');
  fr(c,rx+124,UF-4,12,2,'#a0a4a8');

  /* ── Wall vent ── */
  fbox(c,rx+200,ry+40,24,16,'#a0a8b0');
  for(var vi=0;vi<4;vi++){fr(c,rx+203,ry+44+vi*3,18,1,'#888890');}

  /* ── Ceiling vent ── */
  fbox(c,rx+260,ry-2,30,8,'#909898');
  for(var vi=0;vi<3;vi++){fr(c,rx+263+vi*9,ry,6,5,'#808088');}

  /* ── Extra bathroom items ── */
  /* Toothbrush holder near sink */
  fbox(c,rx+18,UF-48-12,8,10,'#80b0c0');
  fr(c,rx+19,UF-48-16,2,6,'#d0d8e0');fr(c,rx+23,UF-48-18,2,8,'#d8a0a0');
  /* Spray bottle on shelf */
  fbox(c,rx+180,ry+40,5,10,'#88b0d0');fbox(c,rx+181,ry+36,3,5,'#78a0c0');
  /* Slippers on floor */
  fr(c,rx+100,UF-3,6,3,'#88c0b0');fr(c,rx+108,UF-3,6,3,'#88c0b0');
  /* Wall hook with robe */
  fbox(c,rx+140,ry+40,4,3,'#a0a8b0');
  fr(c,rx+136,ry+43,12,30,'#e8e0d8');fr(c,rx+138,ry+45,8,26,'#d8d0c8');
  /* Small shelf with medicines */
  fbox(c,rx+200,ry+80,26,3,'#a0a8b0');
  fbox(c,rx+202,ry+72,5,8,'#d08040');fbox(c,rx+209,ry+70,4,10,'#e0e0e0');
  fbox(c,rx+215,ry+74,6,6,'#a0d0b0');
  /* Plunger in corner */
  fr(c,rx+310,UF-16,4,16,'#a09070');fr(c,rx+308,UF-3,8,4,'#504040');

  /* ── More bathroom wall items ── */
  /* Decorative sea art */
  fbox(c,rx+220,ry+70,30,22,'#d0c8b8');
  fr(c,rx+223,ry+73,24,16,'#90c0d0');
  fr(c,rx+223,ry+81,24,8,'#c0b880');
  fr(c,rx+230,ry+76,8,7,'#e8e0d0');

  /* Wall mounted cabinet (glass doors) */
  box(c,rx+240,ry+50,60,40,'#c0c8cc');
  fbox(c,rx+244,ry+54,24,32,'#d8e0e8');
  fbox(c,rx+272,ry+54,24,32,'#d8e0e8');
  /* Items inside */
  fbox(c,rx+247,ry+58,8,12,'#e8d0c0');fbox(c,rx+257,ry+60,8,10,'#c0e0d0');
  fbox(c,rx+275,ry+56,6,14,'#b0c0e0');fbox(c,rx+283,ry+60,8,10,'#e0c0d0');
  /* Cabinet handles */
  fr(c,rx+266,ry+68,4,4,'#a0a8b0');fr(c,rx+294,ry+68,4,4,'#a0a8b0');

  /* Bright colored bath bomb on tub rim */
  fr(c,rx+180,UF-52-2,6,5,'#d860a0');fr(c,rx+190,UF-52-2,5,5,'#60c0d0');

  /* Candle on shelf */
  fbox(c,rx+150,ry+52,4,8,'#e0d0b0');fr(c,rx+151,ry+50,2,3,'#e0a040');

  /* Small waste bin */
  fbox(c,rx+98,UF-16,10,16,'#b8b0a8');fr(c,rx+100,UF-14,6,12,'#a8a098');
  fbox(c,rx+97,UF-18,12,4,'#c0b8b0');

  /* Bathmat pattern — stripes */
  for(var si=0;si<4;si++){fr(c,rx+147+si*12,UF-4,5,5,'#78b8a8');}
}

/* ═══ LIVING ROOM (lower-left) ═══ */
function drawLiving(c){
  var rx=WP, ry=LY+8;

  /* ── Window ── */
  var wx=rx+16,wy=ry+24,ww=90,wh=72;
  fbox(c,wx-4,wy-4,ww+8,wh+8,'#706858');
  fr(c,wx,wy,ww,wh,'#80a8c8');
  fr(c,wx,wy+wh*0.65|0,ww,wh*0.35|0,'#7098b8');
  /* City buildings in living room window */
  fr(c,wx+2,wy+wh*0.4|0,6,wh*0.28|0,'#5888a8');
  fr(c,wx+12,wy+wh*0.35|0,8,wh*0.33|0,'#5080a0');
  fr(c,wx+24,wy+wh*0.42|0,6,wh*0.26|0,'#5888a8');
  fr(c,wx+34,wy+wh*0.3|0,10,wh*0.38|0,'#4878a0');
  fr(c,wx+50,wy+wh*0.37|0,8,wh*0.31|0,'#5080a0');
  fr(c,wx+62,wy+wh*0.33|0,8,wh*0.35|0,'#5888a8');
  fr(c,wx+74,wy+wh*0.4|0,10,wh*0.28|0,'#5080a0');
  /* Building lights */
  fr(c,wx+14,wy+wh*0.38|0,2,2,'#e0d878');fr(c,wx+36,wy+wh*0.34|0,2,2,'#f0e898');
  fr(c,wx+52,wy+wh*0.42|0,2,2,'#e0d878');fr(c,wx+64,wy+wh*0.37|0,2,2,'#e8e090');
  /* Clouds */
  fr(c,wx+18,wy+10,22,6,'#b0c8e0');
  fr(c,wx+55,wy+18,16,5,'#b0c8e0');
  fr(c,wx+ww/2-1,wy,2,wh,'#706858');
  fr(c,wx,wy+wh/2-1,ww,2,'#706858');
  fbox(c,wx-6,wy+wh,ww+12,5,'#706858');
  /* Curtains */
  fbox(c,wx-16,wy-4,14,wh+32,'#c8b8a0');
  fr(c,wx-14,wy,2,wh+24,'#b8a890');fr(c,wx-10,wy,2,wh+24,'#b8a890');
  fbox(c,wx+ww+2,wy-4,14,wh+32,'#c8b8a0');
  fr(c,wx+ww+4,wy,2,wh+24,'#b8a890');fr(c,wx+ww+10,wy,2,wh+24,'#b8a890');

  /* ── Wall art gallery ── */
  fbox(c,rx+130,ry+26,32,24,'#d0c0b0');
  fr(c,rx+133,ry+29,26,18,'#a8c8d8');
  fr(c,rx+133,ry+39,26,8,'#88b8a0');
  fbox(c,rx+170,ry+22,24,32,'#c0b0a0');
  fr(c,rx+173,ry+25,18,26,'#d0b8a0');
  fr(c,rx+176,ry+28,12,14,'#d8a088');
  fbox(c,rx+200,ry+28,28,22,'#d0c8b8');
  fr(c,rx+203,ry+31,22,16,'#b0a890');

  /* ── TV on wall mount ── */
  var tvx=rx+20,tvy=LF-90;
  /* Wall bracket */
  fbox(c,tvx+16,tvy-8,24,8,'#505050');
  /* TV */
  box(c,tvx,tvy,56,42,'#282830');
  fr(c,tvx+3,tvy+3,50,34,'#181828');
  /* TV stand LED */
  fr(c,tvx+25,tvy+39,6,1,'#40c060');

  /* ── TV cabinet ── */
  box(c,tvx-4,LF-42,64,42,'#605848');
  fbox(c,tvx-1,LF-38,26,14,'#706858');fr(c,tvx+7,LF-34,10,3,'#887868');
  fbox(c,tvx+29,LF-38,30,14,'#706858');fr(c,tvx+39,LF-34,10,3,'#887868');
  fbox(c,tvx-1,LF-22,56,16,'#706858');
  /* Game console */
  fbox(c,tvx+34,LF-20,16,6,'#303038');
  fr(c,tvx+36,LF-19,4,3,'#4060c0');
  /* DVDs/games */
  for(var i=0;i<4;i++){fbox(c,tvx+3+i*7,LF-20,5,12,'#'+(i%2?'5080b0':'c05050'));}

  /* ── Floor lamp ── */
  fbox(c,rx+86,ry+100,3,LF-ry-106,'#605848');
  fbox(c,rx+82,ry+96,11,8,'#d8c8a0');
  fr(c,rx+84,ry+102,7,4,'#e8d8b0');
  fbox(c,rx+84,LF-6,6,6,'#605848');
  c.fillStyle='rgba(255,240,200,0.03)';c.fillRect(rx+70,ry+104,30,60);

  /* ── Sofa (large) ── */
  var sx=rx+110,sy=LF-56;
  /* Back */
  box(c,sx,sy-12,140,16,'#5a3830');
  /* Body */
  box(c,sx,sy+2,140,36,'#5a3830');
  /* Cushions */
  fbox(c,sx+8,sy+6,56,26,'#d898a8');
  fbox(c,sx+68,sy+6,64,26,'#d898a8');
  fr(c,sx+8,sy+6,56,3,'#c08090');
  fr(c,sx+68,sy+6,64,3,'#c08090');
  /* Cushion stitch */
  fr(c,sx+64,sy+8,4,22,'#a87080');
  /* Armrests */
  box(c,sx-8,sy-6,12,44,'#6a4838');
  box(c,sx+136,sy-6,12,44,'#6a4838');
  /* Throw pillows */
  fbox(c,sx+12,sy-8,18,16,'#e8d0b0');
  fbox(c,sx+110,sy-8,18,16,'#a8c8e0');
  fbox(c,sx+60,sy-6,14,12,'#c0e0c0');
  /* Legs */
  fbox(c,sx+4,sy+38,8,18,'#483828');
  fbox(c,sx+128,sy+38,8,18,'#483828');

  /* ── Coffee table ── */
  var cofX=rx+138,cofY=LF-24;
  box(c,cofX,cofY,60,6,'#806848');
  /* Shelf under table */
  fr(c,cofX+4,cofY+14,52,3,'#705838');
  fbox(c,cofX+6,cofY+6,4,20,'#705838');
  fbox(c,cofX+50,cofY+6,4,20,'#705838');
  /* Items on table */
  fbox(c,cofX+8,cofY-6,16,6,'#c04848');
  fbox(c,cofX+10,cofY-5,12,4,'#d06060');
  fbox(c,cofX+28,cofY-8,10,8,'#d8d0c0');
  fr(c,cofX+30,cofY-6,6,4,'#b8a068');
  fbox(c,cofX+42,cofY-4,12,4,'#404848');
  /* Magazines under table */
  fr(c,cofX+12,cofY+15,16,2,'#c0b0a0');
  fr(c,cofX+14,cofY+14,14,2,'#a0b0c0');

  /* ── Bookshelf (tall) ── */
  var bsx=rx+268,bsy=ry+60;
  box(c,bsx,bsy,60,160,'#806848');
  var bkCols=['#c05050','#5080b0','#50a050','#c0a040','#8868a8','#c07040','#a05080'];
  for(var si=0;si<5;si++){
    var ssy=bsy+4+si*31;
    fr(c,bsx+3,ssy+26,54,3,'#806848');
    var bk=bsx+5;
    for(var bi=0;bi<5;bi++){
      var bw=5+(bi*3+si)%5;
      fbox(c,bk,ssy,bw,25,bkCols[(bi+si*2)%7]);
      bk+=bw+1;
    }
    /* Random small item on shelf */
    if(si===1) fbox(c,bsx+44,ssy+10,8,16,'#d8c8a0');
    if(si===3){fr(c,bsx+46,ssy+8,6,8,'#58a058');fr(c,bsx+47,ssy+4,4,6,'#68b068');}
  }

  /* ── Plant ── */
  var px=rx+300,py=LF;
  fbox(c,px,py-32,28,32,'#907050');
  fr(c,px+2,py-30,24,28,'#806040');
  fr(c,px+4,py-30,20,4,'#504030');
  fr(c,px+4,py-50,20,22,'#58a058');
  fr(c,px-4,py-44,14,16,'#48a048');
  fr(c,px+18,py-46,14,16,'#68b068');
  fr(c,px+6,py-58,12,14,'#58b058');
  fr(c,px+12,py-38,4,12,'#388038');
  /* Small flower */
  fr(c,px+8,py-60,4,4,'#e0a0a0');
  fr(c,px+20,py-48,3,3,'#e0c060');

  /* ── Rug ── */
  fr(c,rx+100,LF-5,160,7,'#c8b898');
  fr(c,rx+102,LF-4,156,5,'#b8a888');
  /* Rug geometric pattern */
  for(var i=0;i<7;i++){
    fr(c,rx+110+i*20,LF-4,8,5,(i%2)?'#c8b898':'#a89878');
  }

  /* ── Wall outlet ── */
  fbox(c,rx+98,ry+180,8,10,'#c8c0b8');
  fr(c,rx+100,ry+182,2,2,'#484040');fr(c,rx+102,ry+182,2,2,'#484040');

  /* ── Shoes near door ── */
  fr(c,rx+320,LF-4,8,3,'#a87868');
  fr(c,rx+312,LF-4,7,3,'#6080b0');

  /* ── Extra living room items ── */
  /* Blanket draped on sofa */
  fr(c,rx+230,LF-62,16,28,'#c8d0a8');fr(c,rx+232,LF-60,12,24,'#b8c098');
  /* Remote on coffee table */
  fbox(c,rx+172,LF-27,14,4,'#303038');fr(c,rx+174,LF-26,3,2,'#c04040');
  /* Umbrella stand near door */
  fbox(c,rx+308,LF-26,10,26,'#806858');
  fr(c,rx+310,LF-40,2,16,'#404060');fr(c,rx+314,LF-44,2,20,'#c04040');
  /* Cable box on floor */
  fbox(c,rx+26,LF-10,14,8,'#484848');fr(c,rx+28,LF-9,3,2,'#40c060');fr(c,rx+33,LF-9,3,2,'#4060c0');
  /* Wall switch */
  fbox(c,rx+106,ry+120,6,8,'#c8c0b8');fr(c,rx+107,ry+122,4,2,'#a0a0a0');
  /* Small picture frame on wall */
  fbox(c,rx+240,ry+36,20,16,'#b8a890');
  fr(c,rx+243,ry+39,14,10,'#d0c8b8');fr(c,rx+245,ry+41,10,6,'#a8c8b8');
  /* Cat toy on floor */
  fr(c,rx+200,LF-4,5,3,'#d06060');fr(c,rx+204,LF-5,3,2,'#c8c8c8');
  /* Tissue box on coffee table */
  fbox(c,rx+156,LF-30,10,7,'#e0d8d0');fr(c,rx+159,LF-30,4,2,'#f0e8e0');

  /* ── Bright vinyl record display on wall ── */
  fbox(c,rx+238,ry+66,24,24,'#181818');
  fr(c,rx+240,ry+68,20,20,'#282828');
  fr(c,rx+246,ry+74,8,8,'#d04060');
  fr(c,rx+249,ry+77,2,2,'#e8e0d0');

  /* ── Magazine stack on floor near sofa ── */
  fr(c,rx+108,LF-6,12,4,'#d0b0a0');fr(c,rx+110,LF-8,10,4,'#a0c0d0');
  fr(c,rx+109,LF-10,11,4,'#e0c080');

  /* ── Game controller on sofa ── */
  fbox(c,rx+178,LF-56+6+8,12,6,'#383840');
  fr(c,rx+180,LF-56+6+9,3,2,'#60c060');fr(c,rx+185,LF-56+6+9,3,2,'#c06060');

  /* ── String lights on bookshelf ── */
  var slCols=['#e08040','#e0c040','#e06060','#40c0e0'];
  for(var si=0;si<4;si++){fr(c,rx+272+si*13,ry+58,2,2,slCols[si]);}

  /* ── Coat hook near door with jacket ── */
  fbox(c,rx+316,ry+80,4,3,'#808080');
  fr(c,rx+312,ry+83,12,28,'#385060');fr(c,rx+314,ry+85,8,24,'#405868');

  /* ── Floor cable ── */
  fr(c,rx+86,LF-3,100,1,'#404040');
}

/* ═══ KITCHEN (lower-right) ═══ */
function drawKitchen(c){
  var rx=CX+CW, ry=LY+8;

  /* ── Window ── */
  var wx=rx+228,wy=ry+28,ww=76,wh=64;
  fbox(c,wx-4,wy-4,ww+8,wh+8,'#706858');
  fr(c,wx,wy,ww,wh,'#c8a060');
  fr(c,wx,wy+wh*0.6|0,ww,wh*0.4|0,'#b89050');
  /* Sunset sky gradient */
  fr(c,wx,wy,ww,wh*0.3|0,'#d88860');
  fr(c,wx,wy+wh*0.15|0,ww,wh*0.2|0,'#c8a070');
  /* Building silhouettes */
  fr(c,wx+4,wy+wh*0.35|0,10,wh*0.3|0,'#806040');
  fr(c,wx+18,wy+wh*0.4|0,8,wh*0.25|0,'#907050');
  fr(c,wx+30,wy+wh*0.3|0,6,wh*0.35|0,'#705838');
  fr(c,wx+42,wy+wh*0.38|0,10,wh*0.28|0,'#806040');
  fr(c,wx+56,wy+wh*0.32|0,8,wh*0.33|0,'#907050');
  fr(c,wx+68,wy+wh*0.42|0,6,wh*0.24|0,'#806040');
  /* Sun */
  fr(c,wx+52,wy+6,10,10,'#f0c060');fr(c,wx+54,wy+8,6,6,'#f8d880');
  fr(c,wx+14,wy+10,18,6,'#d8b878');
  fr(c,wx+ww/2-1,wy,2,wh,'#706858');
  fr(c,wx,wy+wh/2-1,ww,2,'#706858');
  fbox(c,wx-6,wy+wh,ww+12,5,'#706858');
  /* Small herb on sill */
  fbox(c,wx+8,wy+wh-7,8,7,'#907050');
  fr(c,wx+9,wy+wh-12,6,7,'#58a858');

  /* ── Upper cabinets (long row) ── */
  box(c,rx+10,ry+44,60,44,'#908068');
  fbox(c,rx+14,ry+48,24,36,'#a09078');fr(c,rx+36,ry+62,4,8,'#b8a888');
  fbox(c,rx+42,ry+48,24,36,'#a09078');fr(c,rx+64,ry+62,4,8,'#b8a888');
  box(c,rx+78,ry+44,60,44,'#908068');
  fbox(c,rx+82,ry+48,24,36,'#a09078');fr(c,rx+104,ry+62,4,8,'#b8a888');
  fbox(c,rx+110,ry+48,24,36,'#a09078');fr(c,rx+132,ry+62,4,8,'#b8a888');
  /* Glass cabinet */
  box(c,rx+146,ry+44,40,44,'#908068');
  fbox(c,rx+150,ry+48,32,36,'#b0b8b0');
  /* Plates visible through glass */
  for(var i=0;i<3;i++){fr(c,rx+154+i*10,ry+56,6,20,'#d8d0c0');}

  /* ── Hanging pots ── */
  fbox(c,rx+200,ry+50,36,3,'#808078');
  fbox(c,rx+202,ry+53,3,12,'#606060');
  fbox(c,rx+198,ry+65,12,10,'#484848');fr(c,rx+200,ry+67,8,6,'#585858');
  fbox(c,rx+216,ry+53,3,10,'#606060');
  fbox(c,rx+212,ry+63,12,8,'#584838');
  fbox(c,rx+228,ry+53,3,14,'#606060');
  fbox(c,rx+224,ry+67,12,10,'#484848');fr(c,rx+226,ry+69,8,6,'#585858');

  /* ── Counter / Stove ── */
  var stx=rx+10,sty=LF-48;
  box(c,stx,sty,140,48,'#908078');
  fbox(c,stx,sty,140,7,'#a89888');
  /* Stove burners */
  fbox(c,stx+10,sty+1,24,5,'#404040');
  fr(c,stx+14,sty+2,4,3,'#505050');fr(c,stx+22,sty+2,4,3,'#505050');
  fbox(c,stx+42,sty+1,24,5,'#404040');
  fr(c,stx+46,sty+2,4,3,'#505050');fr(c,stx+54,sty+2,4,3,'#505050');
  /* Oven door */
  fbox(c,stx+8,sty+16,54,28,'#807068');
  fr(c,stx+12,sty+20,46,20,'#605048');
  fr(c,stx+14,sty+16,36,3,'#989080');
  /* Oven window */
  fr(c,stx+20,sty+24,32,12,'#504040');
  /* Drawers */
  fbox(c,stx+70,sty+12,64,14,'#988878');fr(c,stx+96,sty+17,12,3,'#b8a898');
  fbox(c,stx+70,sty+28,64,16,'#988878');fr(c,stx+96,sty+34,12,3,'#b8a898');

  /* ── Kettle on stove ── */
  fbox(c,stx+14,sty-16,18,16,'#b0b0b8');
  fr(c,stx+16,sty-14,14,12,'#c0c0c8');
  fbox(c,stx+30,sty-12,6,4,'#989898');
  fbox(c,stx+20,sty-20,5,6,'#a0a0a8');

  /* ── Cutting board + knife ── */
  fbox(c,stx+80,sty-8,20,6,'#c0a870');
  fr(c,stx+104,sty-6,14,2,'#a0a0a0');
  fr(c,stx+102,sty-8,4,4,'#605040');

  /* ── Sink ── */
  box(c,rx+160,sty,44,48,'#a09888');
  fbox(c,rx+160,sty,44,7,'#b0a898');
  fr(c,rx+166,sty+10,32,20,'#888078');
  fbox(c,rx+178,sty-14,4,16,'#a0aab0');
  fbox(c,rx+174,sty-16,12,4,'#a8b4b8');
  /* Dish rack */
  fbox(c,rx+208,sty-2,20,7,'#a0a8a0');
  for(var d=0;d<3;d++){fr(c,rx+211+d*6,sty-8,2,8,'#d8d0c0');}

  /* ── Fridge (tall) ── */
  var fx=rx+290,fy=LF-100;
  box(c,fx,fy,48,100,'#d0d4d8');
  fbox(c,fx+3,fy+3,42,36,'#d8dce0');
  fr(c,fx+44,fy+14,4,12,'#b8bcc0');
  fbox(c,fx+3,fy+42,42,54,'#d8dce0');
  fr(c,fx+44,fy+58,4,18,'#b8bcc0');
  fr(c,fx+3,fy+39,42,3,shade('#d0d4d8',-20));
  /* Fridge magnets */
  fr(c,fx+8,fy+46,6,6,'#d06060');
  fr(c,fx+18,fy+48,8,4,'#60a060');
  fr(c,fx+30,fy+44,4,8,'#6080c0');
  /* Photo on fridge */
  fr(c,fx+10,fy+56,12,10,'#e8e0d0');
  fr(c,fx+12,fy+58,8,6,'#b0d0e0');

  /* ── Table + Chairs ── */
  var tbx=rx+166,tby=LF-40;
  box(c,tbx,tby,68,7,'#806848');
  fbox(c,tbx+6,tby+7,4,33,'#705838');
  fbox(c,tbx+58,tby+7,4,33,'#705838');
  /* Chair left */
  fbox(c,tbx-18,LF-34,18,5,'#6a5848');
  fbox(c,tbx-16,LF-62,14,30,'#907858');
  fbox(c,tbx-16,LF-29,3,29,'#5a4838');
  fbox(c,tbx-3,LF-29,3,29,'#5a4838');
  /* Chair right */
  fbox(c,tbx+68,LF-34,18,5,'#6a5848');
  fbox(c,tbx+70,LF-62,14,30,'#907858');
  fbox(c,tbx+70,LF-29,3,29,'#5a4838');
  fbox(c,tbx+83,LF-29,3,29,'#5a4838');
  /* Fruit bowl */
  fbox(c,tbx+20,tby-12,28,12,'#d8c8a8');
  fr(c,tbx+24,tby-14,7,5,'#d05040');
  fr(c,tbx+33,tby-14,7,5,'#d0b030');
  fr(c,tbx+28,tby-18,6,5,'#50a050');
  /* Salt & pepper */
  fbox(c,tbx+52,tby-8,5,7,'#e8e0d0');
  fbox(c,tbx+58,tby-8,5,7,'#404040');

  /* ── Trash can ── */
  fbox(c,rx+250,LF-30,18,30,'#707070');
  fr(c,rx+252,LF-28,14,26,'#606060');
  fbox(c,rx+249,LF-32,20,4,'#787878');

  /* ── Wall clock ── */
  fbox(c,rx+260,ry+32,18,18,'#d8d0c0');
  fr(c,rx+262,ry+34,14,14,'#e8e0d8');
  fr(c,rx+268,ry+36,2,7,'#404040');
  fr(c,rx+268,ry+41,5,2,'#404040');

  /* ── Microwave on counter ── */
  fbox(c,stx+100,sty-18,28,16,'#c8c8d0');
  fr(c,stx+103,sty-16,18,10,'#384038');
  fbox(c,stx+123,sty-14,4,6,'#a8a8b0');

  /* ── Spice rack ── */
  fbox(c,rx+148,ry+94,30,4,'#706858');
  for(var i=0;i<4;i++){fbox(c,rx+150+i*7,ry+84,5,10,['#c86040','#a0a040','#c0a050','#608040'][i]);}

  /* ── Extra kitchen items ── */
  /* Coffee maker on counter */
  fbox(c,stx+56,sty-18,14,16,'#303030');fbox(c,stx+58,sty-14,10,8,'#484848');
  fr(c,stx+60,sty-12,6,4,'#908878');fbox(c,stx+67,sty-10,4,8,'#c8c8d0');
  /* Dish towel on oven handle */
  fr(c,stx+14,sty+13,28,3,'#d8d0a0');
  /* Recipe book on counter */
  fbox(c,stx+130,sty-10,8,10,'#c04848');fr(c,stx+131,sty-9,6,8,'#d06060');
  /* Napkin holder on table */
  fbox(c,tbx+8,tby-8,6,8,'#a09080');fr(c,tbx+9,tby-6,4,5,'#e8e0d8');
  /* Fridge magnets — more */
  fr(c,fx+22,fy+50,5,5,'#d0a040');fr(c,fx+34,fy+54,4,4,'#a060b0');
  /* Calendar on fridge */
  fr(c,fx+6,fy+10,14,10,'#e8e0d0');fr(c,fx+8,fy+10,10,2,'#d06060');
  fr(c,fx+8,fy+14,3,2,'#404040');fr(c,fx+13,fy+14,3,2,'#404040');
  /* Mop in corner */
  fr(c,rx+320,LF-40,2,40,'#a09070');fr(c,rx+316,LF-6,10,6,'#a0a8a0');
  /* Floor mat */
  fr(c,rx+160,LF-4,80,5,'#b8a888');fr(c,rx+162,LF-3,76,3,'#a89878');
  /* Wall hooks with pots/pans */
  fbox(c,rx+188,ry+30,3,3,'#a0a0a0');fbox(c,rx+200,ry+30,3,3,'#a0a0a0');
  fbox(c,rx+186,ry+33,8,6,'#686060');fbox(c,rx+198,ry+33,8,6,'#806048');

  /* ── Bright red apron on wall hook ── */
  fbox(c,rx+8,ry+80,3,3,'#808080');
  fr(c,rx+4,ry+83,12,24,'#d04040');fr(c,rx+6,ry+85,8,20,'#c03838');
  fr(c,rx+6,ry+83,8,3,'#b03030');

  /* ── Paper towel holder on wall ── */
  fbox(c,rx+145,ry+120,3,16,'#a0a0a0');
  fr(c,rx+140,ry+118,12,10,'#e8e0d0');

  /* ── Grocery bag on floor ── */
  fbox(c,rx+240,LF-18,12,18,'#c8b890');fr(c,rx+242,LF-16,8,14,'#b8a880');
  /* Veggies sticking out */
  fr(c,rx+243,LF-22,3,6,'#40a040');fr(c,rx+247,LF-20,3,4,'#d08030');

  /* ── Magnetic knife strip on wall ── */
  fbox(c,rx+60,ry+100,40,3,'#505050');
  fr(c,rx+64,ry+94,2,8,'#a0a0a8');fr(c,rx+74,ry+92,2,10,'#a0a0a8');
  fr(c,rx+84,ry+96,2,6,'#a0a0a8');fr(c,rx+92,ry+93,2,9,'#a0a0a8');

  /* ── Cat food bowl on floor ── */
  fbox(c,rx+28,LF-5,14,5,'#d8c8b0');
  fr(c,rx+30,LF-4,10,3,'#a88060');
  fbox(c,rx+46,LF-5,10,5,'#90b0d0');
  fr(c,rx+48,LF-4,6,3,'#6098c8');

  /* ── Bright fridge sticker ── */
  fr(c,rx+290+8,LF-100+70,10,10,'#e0c040');
  fr(c,rx+290+10,LF-100+72,6,6,'#f0d860');

  /* ── Oven mitt hung on stove ── */
  fbox(c,rx+10+140,LF-48+3,8,14,'#d06040');fr(c,rx+10+142,LF-48+5,4,10,'#c05838');
}

/* ═══ ROOM PIPES & CEILING INFRASTRUCTURE ═══ */
function drawRoomInfra(c){
  /* — Bedroom: exposed pipe near ceiling — */
  fr(c,WP+4,UY+12,CX-WP-8,3,'#989088');
  fr(c,WP+4,UY+12,CX-WP-8,1,'#a8a098');
  /* Pipe brackets */
  for(var bx=WP+30;bx<CX-10;bx+=60){fbox(c,bx,UY+10,4,7,'#888078');}
  /* Sprinkler head */
  fbox(c,WP+200,UY+12,6,6,'#b0a8a0');fr(c,WP+201,UY+17,4,2,'#c0b8b0');
  /* Smoke detector */
  fbox(c,WP+60,UY+4,10,6,'#d8d0c8');fr(c,WP+63,UY+6,4,2,'#c04040');

  /* — Bathroom: water pipe + drain pipe — */
  fr(c,CX+CW+4,UY+14,W-WP-CX-CW-8,3,'#708878');
  fr(c,CX+CW+4,UY+14,W-WP-CX-CW-8,1,'#80a088');
  for(var bx=CX+CW+40;bx<W-WP-10;bx+=55){fbox(c,bx,UY+12,4,7,'#688070');}
  /* Drain pipe going down wall */
  fr(c,W-WP-18,UY+14,4,UH-20,'#708070');
  fr(c,W-WP-18,UY+14,4,1,'#80a080');
  /* Pipe joints */
  fbox(c,W-WP-20,UY+80,8,6,'#809080');
  fbox(c,W-WP-20,UY+200,8,6,'#809080');

  /* — Living room: cable tray near ceiling — */
  fr(c,WP+4,LY+10,CX-WP-8,4,'#787070');
  fr(c,WP+4,LY+10,CX-WP-8,1,'#888078');
  for(var bx=WP+24;bx<CX-10;bx+=50){
    fbox(c,bx,LY+8,3,8,'#686060');
    /* Cables hanging */
    fr(c,bx+1,LY+14,1,6,'#404040');
  }
  /* Vent/AC unit */
  fbox(c,WP+280,LY+4,40,12,'#a0a098');
  for(var vi=0;vi<5;vi++){fr(c,WP+284+vi*7,LY+7,4,5,'#909088');}

  /* — Kitchen: exhaust pipe + cable tray — */
  fr(c,CX+CW+4,LY+12,W-WP-CX-CW-8,3,'#888078');
  for(var bx=CX+CW+30;bx<W-WP-10;bx+=65){fbox(c,bx,LY+10,4,7,'#787070');}
  /* Range hood exhaust duct */
  fbox(c,CX+CW+24,LY+4,30,10,'#a09888');
  fr(c,CX+CW+26,LY+6,26,6,'#908878');

  /* ═══ WALL WEAR / GRIME SPOTS ═══ */
  /* Subtle stain marks */
  c.fillStyle='rgba(0,0,0,0.02)';
  c.fillRect(WP+60,UY+180,30,40);
  c.fillRect(CX+CW+80,UY+120,25,35);
  c.fillRect(WP+200,LY+100,20,30);
  c.fillRect(CX+CW+200,LY+150,30,25);
  /* Floor wear near doorways */
  c.fillStyle='rgba(0,0,0,0.03)';
  c.fillRect(CX-20,UF-6,CW+40,8);
  c.fillRect(CX-20,LF-6,CW+40,8);
  c.fillRect(STX-10,UF-6,STW+20,8);
  c.fillRect(STX-10,LF-6,STW+20,8);
  /* Baseboard scuff marks */
  c.fillStyle='rgba(0,0,0,0.04)';
  c.fillRect(WP+100,UF-8,12,6);
  c.fillRect(WP+240,LF-8,8,6);
  c.fillRect(CX+CW+120,UF-8,10,6);
  c.fillRect(CX+CW+180,LF-8,14,6);
}

/* ═══ BUILD BACKGROUND ═══ */
function buildBackground(){
  drawStructure(bg);
  drawBedroom(bg);
  drawBathroom(bg);
  drawLiving(bg);
  drawKitchen(bg);
  drawRoomInfra(bg);
  bgOK=true;
}

/* ═══ CHARACTER STATE ═══ */
var ch={
  x:130,y:UF,dir:0,frame:0,fTimer:0,
  state:'idle',sTimer:0,idleWait:3,
  path:[],pathIdx:0,moving:false,speed:80,
  currentRoom:'bedroom',
  bubble:'',jumpOff:0,jumpT:0,
  pose:'stand',lastAct:Date.now()
};

function getRoomAt(x,y){
  if(y<=MY+MID/2) return x<CX+CW/2?'bedroom':'bathroom';
  return x<CX+CW/2?'living':'kitchen';
}

var doorUL={x:CX-15,y:UF}, doorUR={x:CX+CW+15,y:UF};
var doorLL={x:CX-15,y:LF}, doorLR={x:CX+CW+15,y:LF};
var stairT={x:STX+STW/2,y:UF}, stairB={x:STX+STW/2,y:LF};
var stairMid={x:STX+STW*0.6,y:(UF+LF)/2};

var PT={};
PT.bedroom_bathroom=[doorUL,doorUR];
PT.bathroom_bedroom=[doorUR,doorUL];
PT.living_kitchen=[doorLL,doorLR];
PT.kitchen_living=[doorLR,doorLL];
PT.bedroom_living=[stairT,stairMid,stairB];
PT.living_bedroom=[stairB,stairMid,stairT];
PT.bedroom_kitchen=[stairT,stairMid,stairB,doorLL,doorLR];
PT.kitchen_bedroom=[doorLR,doorLL,stairB,stairMid,stairT];
PT.bathroom_living=[doorUR,doorUL,stairT,stairMid,stairB];
PT.living_bathroom=[stairB,stairMid,stairT,doorUL,doorUR];
PT.bathroom_kitchen=[doorUR,doorUL,stairT,stairMid,stairB,doorLL,doorLR];
PT.kitchen_bathroom=[doorLR,doorLL,stairB,stairMid,stairT,doorUL,doorUR];

function buildPath(room,dx,dy){
  var from=ch.currentRoom;
  if(from===room) return [{x:dx,y:dy}];
  var wp=PT[from+'_'+room]||[];
  var pts=[];
  for(var i=0;i<wp.length;i++) pts.push({x:wp[i].x,y:wp[i].y});
  pts.push({x:dx,y:dy});
  return pts;
}

function randomIdleTarget(){
  var rooms=['bedroom','bathroom','living','kitchen'];
  var room=rooms[Math.floor(Math.random()*rooms.length)];
  var tx,ty;
  if(room==='bedroom'){tx=30+Math.random()*280;ty=UF;}
  else if(room==='bathroom'){tx=CX+CW+30+Math.random()*260;ty=UF;}
  else if(room==='living'){tx=30+Math.random()*280;ty=LF;}
  else{tx=CX+CW+30+Math.random()*260;ty=LF;}
  return {room:room,x:tx,y:ty};
}

/* ═══ CHARACTER SPRITES — 14w×30h side-view ═══ */
var SK='#ffdcc8',SKS='#f0c8b0',HR='#3a2820',HRH='#5a4838';
var EY='#2858a0',EYH='#fff',BL='#f0a8a0',LI='#e08888',RB='#f08098';
var SW='#f0a0b8',SWH='#f8b8c8',SWD='#d88898';
var S2='#6888b8',S2D='#5070a0',SHO='#d88898';

function drawUpperBody(ox,oy,dir){
  function p(lx,ly,w,h){if(dir===1)ctx.fillRect(ox+14-lx-w,oy+ly,w,h);else ctx.fillRect(ox+lx,oy+ly,w,h);}
  ctx.fillStyle=HR;p(1,0,8,5);p(0,5,4,18);
  ctx.fillStyle=HRH;p(3,1,4,2);
  ctx.fillStyle=SK;p(4,2,8,9);
  ctx.fillStyle=SKS;p(4,9,8,2);
  ctx.fillStyle=EY;p(9,4,2,2);
  ctx.fillStyle=EYH;p(10,4,1,1);
  ctx.fillStyle=BL;p(9,7,2,1);
  ctx.fillStyle=LI;p(8,8,2,1);
  ctx.fillStyle=RB;p(2,1,4,3);
  ctx.fillStyle=SW;p(3,11,9,7);
  ctx.fillStyle=SWH;p(8,12,4,2);
  ctx.fillStyle=SWD;p(3,11,9,1);
  ctx.fillStyle=SW;p(11,13,3,4);
  ctx.fillStyle=SK;p(11,17,3,2);
  ctx.fillStyle=S2;p(3,18,9,4);
  ctx.fillStyle=S2D;p(3,18,9,1);
}

function drawCharStand(x,y,dir){
  var ox=x-7,oy=y-30+ch.jumpOff;
  ctx.fillStyle='rgba(0,0,0,0.18)';ctx.fillRect(x-6,y-2,12,5);
  drawUpperBody(ox,oy,dir);
  function p(lx,ly,w,h){if(dir===1)ctx.fillRect(ox+14-lx-w,oy+ly,w,h);else ctx.fillRect(ox+lx,oy+ly,w,h);}
  ctx.fillStyle=SK;p(4,22,3,5);p(8,22,3,5);
  ctx.fillStyle=SHO;p(4,27,4,3);p(8,27,4,3);
}
function drawCharWalk(x,y,dir,frame){
  var ox=x-7,oy=y-30+ch.jumpOff;
  ctx.fillStyle='rgba(0,0,0,0.18)';ctx.fillRect(x-6,y-2,12,5);
  drawUpperBody(ox,oy,dir);
  function p(lx,ly,w,h){if(dir===1)ctx.fillRect(ox+14-lx-w,oy+ly,w,h);else ctx.fillRect(ox+lx,oy+ly,w,h);}
  ctx.fillStyle=SK;
  if(frame===0){p(3,22,3,5);p(9,22,3,4);}else{p(5,22,3,4);p(7,22,3,5);}
  ctx.fillStyle=SHO;
  if(frame===0){p(2,27,4,3);p(9,26,4,3);}else{p(5,26,4,3);p(6,27,4,3);}
}
function drawCharSitting(x,y,dir){
  var ox=x-7,oy=y-26;
  ctx.fillStyle='rgba(0,0,0,0.06)';ctx.fillRect(x-4,y-1,8,3);
  drawUpperBody(ox,oy,dir);
  function p(lx,ly,w,h){if(dir===1)ctx.fillRect(ox+14-lx-w,oy+ly,w,h);else ctx.fillRect(ox+lx,oy+ly,w,h);}
  ctx.fillStyle=S2;p(3,22,10,3);
  ctx.fillStyle=SK;p(12,22,2,5);
  ctx.fillStyle=SHO;p(12,27,3,2);
}
function drawCharThinking(x,y,dir){
  var ox=x-7,oy=y-26;
  ctx.fillStyle='rgba(0,0,0,0.06)';ctx.fillRect(x-4,y-1,8,3);
  function p(lx,ly,w,h){if(dir===1)ctx.fillRect(ox+14-lx-w,oy+ly,w,h);else ctx.fillRect(ox+lx,oy+ly,w,h);}
  ctx.fillStyle=HR;p(1,0,8,5);p(0,5,4,18);ctx.fillStyle=HRH;p(3,1,4,2);
  ctx.fillStyle=SK;p(4,2,8,9);ctx.fillStyle=SKS;p(4,9,8,2);
  ctx.fillStyle=EY;p(9,5,2,1);ctx.fillStyle=BL;p(9,7,2,1);ctx.fillStyle=LI;p(8,8,2,1);
  ctx.fillStyle=RB;p(2,1,4,3);
  ctx.fillStyle=SW;p(3,11,9,7);ctx.fillStyle=SWH;p(8,12,4,2);ctx.fillStyle=SWD;p(3,11,9,1);
  ctx.fillStyle=SW;p(9,11,5,3);ctx.fillStyle=SK;p(10,8,3,3);
  ctx.fillStyle=S2;p(3,18,9,4);ctx.fillStyle=S2D;p(3,18,9,1);
  ctx.fillStyle=S2;p(3,22,10,3);ctx.fillStyle=SK;p(12,22,2,5);ctx.fillStyle=SHO;p(12,27,3,2);
}
function drawCharSleeping(x,y){
  var ox=x-15,oy=y-8;
  ctx.fillStyle='#d898a8';ctx.fillRect(ox-10,oy+4,30,12);
  ctx.fillStyle='#c08090';ctx.fillRect(ox-10,oy+4,30,2);
  ctx.fillStyle='#e0a8b8';ctx.fillRect(ox-6,oy+8,10,4);ctx.fillRect(ox+10,oy+8,10,4);
  ctx.fillStyle=HR;ctx.fillRect(ox+20,oy-4,10,8);ctx.fillRect(ox+18,oy+2,14,6);
  ctx.fillStyle=HRH;ctx.fillRect(ox+22,oy-2,4,2);
  ctx.fillStyle=SK;ctx.fillRect(ox+22,oy-2,8,6);
  ctx.fillStyle=EY;ctx.fillRect(ox+27,oy,2,1);
  ctx.fillStyle=BL;ctx.fillRect(ox+27,oy+2,2,1);
  ctx.fillStyle=LI;ctx.fillRect(ox+27,oy+3,1,1);
}
function drawCharReading(x,y,dir){
  var ox=x-7,oy=y-26;
  ctx.fillStyle='rgba(0,0,0,0.06)';ctx.fillRect(x-4,y-1,8,3);
  function p(lx,ly,w,h){if(dir===1)ctx.fillRect(ox+14-lx-w,oy+ly,w,h);else ctx.fillRect(ox+lx,oy+ly,w,h);}
  ctx.fillStyle=HR;p(1,0,8,5);p(0,5,4,18);ctx.fillStyle=HRH;p(3,1,4,2);
  ctx.fillStyle=SK;p(4,2,8,9);ctx.fillStyle=SKS;p(4,9,8,2);
  ctx.fillStyle=EY;p(8,5,2,2);ctx.fillStyle=EYH;p(9,5,1,1);
  ctx.fillStyle=BL;p(9,7,2,1);ctx.fillStyle=LI;p(8,8,2,1);ctx.fillStyle=RB;p(2,1,4,3);
  ctx.fillStyle=SW;p(3,11,9,7);ctx.fillStyle=SWH;p(8,12,4,2);ctx.fillStyle=SWD;p(3,11,9,1);
  ctx.fillStyle=SW;p(10,13,4,5);
  ctx.fillStyle='#c04848';p(12,13,6,8);ctx.fillStyle='#d06060';p(13,14,4,6);
  ctx.fillStyle=SK;p(11,18,2,2);
  ctx.fillStyle=S2;p(3,18,9,4);ctx.fillStyle=S2D;p(3,18,9,1);
  ctx.fillStyle=S2;p(3,22,10,3);ctx.fillStyle=SK;p(12,22,2,5);ctx.fillStyle=SHO;p(12,27,3,2);
}
function drawChar(){
  var CS=1.5; /* Character scale factor */
  var ix=Math.round(ch.x),iy=Math.round(ch.y)+ch.jumpOff;
  /* Pulsing glow indicator (drawn at normal scale) */
  if(ch.pose!=='sleep'){
    var gA=0.06+Math.sin(Date.now()/400)*0.03;
    ctx.fillStyle='rgba(240,160,180,'+gA+')';
    ctx.fillRect(ix-14,iy-50,28,54);
    /* Indicator dot */
    ctx.fillStyle='rgba(255,100,160,0.65)';
    ctx.fillRect(ix-2,iy-54,4,4);
    ctx.fillStyle='rgba(255,100,160,0.4)';
    ctx.fillRect(ix-1,iy-56,2,2);
  }else{
    ctx.fillStyle='rgba(240,160,180,0.05)';
    ctx.fillRect(ix-22,iy-6,44,18);
  }
  /* Scale up character sprite */
  ctx.save();
  ctx.imageSmoothingEnabled=false;
  ctx.translate(ch.x,ch.y);
  ctx.scale(CS,CS);
  ctx.translate(-ch.x,-ch.y);
  switch(ch.pose){
    case'walk':drawCharWalk(ch.x,ch.y,ch.dir,ch.frame);break;
    case'sit':drawCharSitting(ch.x,ch.y,ch.dir);break;
    case'sitThink':drawCharThinking(ch.x,ch.y,ch.dir);break;
    case'sleep':drawCharSleeping(ch.x,ch.y);break;
    case'readSofa':drawCharReading(ch.x,ch.y,ch.dir);break;
    default:drawCharStand(ch.x,ch.y,ch.dir);break;
  }
  ctx.restore();
}

/* ═══ CAT ═══ */
var cat={x:220,y:LF,dir:0,frame:0,fTimer:0,moving:false,sleepT:0,state:'idle',waitT:3,tx:220,ty:LF};
function drawCat(x,y,dir,sleeping){
  var ox=Math.round(x)-6,oy=Math.round(y)-8;
  if(sleeping){
    ctx.fillStyle='#f0e0c8';ctx.fillRect(ox,oy+2,12,6);
    ctx.fillStyle='#d8c8a8';ctx.fillRect(ox+2,oy+3,8,1);
    ctx.fillStyle='#e8d0b0';ctx.fillRect(ox+10,oy+3,4,3);
    ctx.fillStyle='#f0c8b0';ctx.fillRect(ox,oy,3,3);ctx.fillRect(ox+4,oy,3,3);
    return;
  }
  ctx.fillStyle='#f0e0c8';
  if(dir===0){
    ctx.fillRect(ox+2,oy,8,5);ctx.fillRect(ox+3,oy+5,7,6);
    ctx.fillStyle='#f0c8b0';ctx.fillRect(ox+2,oy-2,3,3);ctx.fillRect(ox+7,oy-2,3,3);
    ctx.fillStyle='#50a050';ctx.fillRect(ox+7,oy+1,2,1);
    ctx.fillStyle='#e0a0a0';ctx.fillRect(ox+8,oy+3,1,1);
    ctx.fillStyle='#e8d0b0';ctx.fillRect(ox,oy+7,3,2);
    ctx.fillStyle='#d8c8a8';ctx.fillRect(ox+4,oy+7,6,1);
  }else{
    ctx.fillRect(ox+2,oy,8,5);ctx.fillRect(ox+2,oy+5,7,6);
    ctx.fillStyle='#f0c8b0';ctx.fillRect(ox+2,oy-2,3,3);ctx.fillRect(ox+7,oy-2,3,3);
    ctx.fillStyle='#50a050';ctx.fillRect(ox+3,oy+1,2,1);
    ctx.fillStyle='#e0a0a0';ctx.fillRect(ox+3,oy+3,1,1);
    ctx.fillStyle='#e8d0b0';ctx.fillRect(ox+9,oy+7,3,2);
    ctx.fillStyle='#d8c8a8';ctx.fillRect(ox+2,oy+7,6,1);
  }
}

/* ═══ SPEECH BUBBLE ═══ */
function drawBubble(x,y){
  if(!ch.bubble)return;
  var bw=40,bh=22;
  var bx=Math.round(x)-bw/2, by=Math.round(y)-44+ch.jumpOff;
  if(ch.pose==='sleep'){bx=x+20;by=y-18;}
  ctx.fillStyle='#fffdf8';
  ctx.fillRect(bx+2,by,bw-4,bh);ctx.fillRect(bx,by+2,bw,bh-4);ctx.fillRect(bx+1,by+1,bw-2,bh-2);
  ctx.fillRect(bx+bw/2-2,by+bh,4,4);ctx.fillRect(bx+bw/2-1,by+bh+3,2,2);
  ctx.fillStyle='#d8d0c8';
  ctx.fillRect(bx+2,by,bw-4,1);ctx.fillRect(bx+2,by+bh-1,bw-4,1);
  ctx.fillRect(bx,by+2,1,bh-4);ctx.fillRect(bx+bw-1,by+2,1,bh-4);
  ctx.fillRect(bx+1,by+1,1,1);ctx.fillRect(bx+bw-2,by+1,1,1);
  ctx.fillRect(bx+1,by+bh-2,1,1);ctx.fillRect(bx+bw-2,by+bh-2,1,1);
  ctx.fillStyle='#3a3a3a';ctx.font='bold 14px monospace';ctx.textAlign='center';
  ctx.fillText(ch.bubble,bx+bw/2,by+16);
}

/* ═══ ENVIRONMENT ANIMATIONS ═══ */
var envT=0;
function drawEnvAnimations(){
  envT+=1/60;
  /* Monitor code scroll */
  var mmx=WP+20+18+3,mmy=UF-58-48+3,mmw=50,mmh=34;
  ctx.fillStyle='#1a3040';ctx.fillRect(mmx,mmy,mmw,mmh);
  var lc=['#70b890','#90b0d0','#d0b070','#b090c0','#90b0d0','#d09090','#70b890','#b0b0d0'];
  for(var i=0;i<8;i++){
    var ly=mmy+2+i*4-((envT*8)%8);
    if(ly>=mmy&&ly<mmy+mmh-2){ctx.fillStyle=lc[i];ctx.fillRect(mmx+2,ly,8+((i*17)%28),2);}
  }
  if(Math.floor(envT*2)%2===0){ctx.fillStyle='#e0e0e0';ctx.fillRect(mmx+14,mmy+mmh-6,3,3);}

  /* TV — animated channel content */
  var tvsx=WP+20+3,tvsy=LF-90+3;
  var tvPhase=Math.floor(envT/4)%3;
  if(tvPhase===0){
    /* Static */
    for(var i=0;i<40;i++){
      var tx=tvsx+Math.random()*48,ty=tvsy+Math.random()*32;
      var g=Math.floor(Math.random()*60)+20;
      ctx.fillStyle='rgb('+g+','+g+','+(g+20)+')';
      ctx.fillRect(tx,ty,2+Math.random()*4,2);
    }
  }else if(tvPhase===1){
    /* Blue screen with scrolling bars */
    ctx.fillStyle='#182848';ctx.fillRect(tvsx,tvsy,50,34);
    var barY=tvsy+(envT*30)%34;
    ctx.fillStyle='rgba(80,120,200,0.3)';ctx.fillRect(tvsx,barY,50,6);
    ctx.fillStyle='#90b0d0';ctx.fillRect(tvsx+8,tvsy+8,34,4);
    ctx.fillStyle='#70a0c0';ctx.fillRect(tvsx+12,tvsy+16,26,3);
  }else{
    /* Color bars */
    var cols=['#c03030','#30c030','#3030c0','#c0c030','#c030c0','#30c0c0'];
    for(var i=0;i<6;i++){ctx.fillStyle=cols[i];ctx.fillRect(tvsx+i*8+1,tvsy,8,34);}
  }
  /* TV glow on nearby wall */
  var tvGlowA=0.03+Math.sin(envT*2)*0.01;
  ctx.fillStyle='rgba(100,140,200,'+tvGlowA+')';
  ctx.fillRect(WP+10,LF-130,70,80);

  /* Kettle steam */
  var kx=CX+CW+10+20,ky=LF-48-20;
  for(var i=0;i<4;i++){
    var sx2=kx+Math.sin(envT*3+i*1.7)*4;
    var sy2=ky-4-i*5-(envT*12+i*4)%22;
    var sa=Math.max(0,0.18-i*0.04);
    ctx.fillStyle='rgba(200,200,220,'+sa+')';ctx.fillRect(sx2,sy2,3,3);
  }

  /* Water drip */
  if(Math.floor(envT*0.5)%3===0){
    var dy2=LF-48+10+((envT*20)%14);
    ctx.fillStyle='rgba(100,160,220,0.3)';ctx.fillRect(CX+CW+178,dy2,2,3);
  }

  /* Lamp flicker (bedroom nightstand) */
  var lampA=0.04+Math.sin(envT*6)*0.01;
  ctx.fillStyle='rgba(255,240,200,'+lampA+')';
  ctx.fillRect(WP+170-32-15,UF-36-34,58,55);

  /* Clock second hand pulse (bedroom) */
  if(Math.floor(envT)%2===0){
    ctx.fillStyle='#d04040';
    ctx.fillRect(WP+270,UY+8+40,1,1);
  }

  /* Living room lamp glow */
  var lampA2=0.02+Math.sin(envT*4)*0.008;
  ctx.fillStyle='rgba(255,240,200,'+lampA2+')';
  ctx.fillRect(WP+70,LY+8+104,30,60);

  /* Fridge hum indicator (tiny LED blink) */
  if(Math.floor(envT*3)%4!==0){
    ctx.fillStyle='#40c060';ctx.fillRect(CX+CW+290+28,LF-100+9,2,2);
  }

  /* AC vent air shimmer (living room) */
  for(var i=0;i<3;i++){
    var ax=WP+284+i*12+Math.sin(envT*5+i)*3;
    var ay=LY+18+i*4+(envT*8+i*6)%16;
    ctx.fillStyle='rgba(180,200,220,0.04)';ctx.fillRect(ax,ay,6,2);
  }

  /* Bathroom drain pipe drip */
  if(Math.floor(envT*0.7)%4===0){
    var dpy=UY+200+((envT*15)%80);
    if(dpy<MY-10){ctx.fillStyle='rgba(80,140,100,0.15)';ctx.fillRect(W-WP-17,dpy,2,4);}
  }

  /* Fairy lights twinkle (bedroom) */
  var flCols2=['#e06060','#e0c040','#60c060','#6080e0','#e060a0'];
  for(var fi=0;fi<8;fi++){
    var fBright=0.4+Math.sin(envT*3+fi*1.2)*0.4;
    ctx.fillStyle='rgba(255,255,255,'+Math.max(0,fBright*0.1)+')';
    ctx.fillRect(WP+176+fi*15-1,UY+8+56-1,5,5);
  }

  /* Monitor screen glow on desk (bedroom) */
  var mGlow=0.03+Math.sin(envT*1.5)*0.01;
  ctx.fillStyle='rgba(60,120,160,'+mGlow+')';
  ctx.fillRect(WP+28,UF-58-6,80,12);

  /* LED strip pulse (bedroom) */
  var ledA=0.15+Math.sin(envT*2)*0.1;
  ctx.fillStyle='rgba(100,60,200,'+ledA+')';
  ctx.fillRect(WP+36,UF-58-2,50,3);
}

/* ═══ STATE MACHINE ═══ */
function updateLabel(t){var el=document.querySelector('.pixel-room-status');if(el)el.textContent='[ '+t+' ]';}
function setState(s){
  ch.state=s;ch.sTimer=0;ch.bubble='';ch.jumpOff=0;ch.jumpT=0;ch.lastAct=Date.now();
  switch(s){
    case'idle':ch.pose='stand';ch.idleWait=2+Math.random()*3;
      var t=randomIdleTarget();ch.path=buildPath(t.room,t.x,t.y);ch.pathIdx=0;ch.moving=true;
      updateLabel('idle...');break;
    case'at_computer':ch.path=buildPath('bedroom',120,UF);ch.pathIdx=0;ch.moving=true;updateLabel('at computer');break;
    case'thinking':ch.path=buildPath('bedroom',120,UF);ch.pathIdx=0;ch.moving=true;ch.bubble='...';updateLabel('thinking...');break;
    case'sleeping':ch.path=buildPath('bedroom',260,UF);ch.pathIdx=0;ch.moving=true;ch.bubble='zzz';updateLabel('sleeping...');break;
    case'reading':ch.path=buildPath('living',180,LF);ch.pathIdx=0;ch.moving=true;updateLabel('reading...');break;
    case'excited':ch.pose='stand';ch.jumpT=0;ch.bubble='!';updateLabel('excited!');break;
  }
}

/* ═══ UPDATE ═══ */
function update(dt){
  ch.sTimer+=dt;
  if(ch.moving&&ch.path.length>0){
    var tgt=ch.path[ch.pathIdx];
    var dx=tgt.x-ch.x,dy=tgt.y-ch.y,dist=Math.sqrt(dx*dx+dy*dy);
    if(dist<3){
      ch.x=tgt.x;ch.y=tgt.y;ch.currentRoom=getRoomAt(ch.x,ch.y);ch.pathIdx++;
      if(ch.pathIdx>=ch.path.length){
        ch.moving=false;ch.pose='stand';ch.frame=0;
        if(ch.state==='at_computer'){ch.pose='sit';ch.dir=0;ch.y=UF-20;}
        else if(ch.state==='thinking'){ch.pose='sitThink';ch.dir=0;ch.y=UF-20;}
        else if(ch.state==='sleeping'){ch.x=260;ch.y=UF-18;ch.pose='sleep';}
        else if(ch.state==='reading'){ch.y=LF-16;ch.pose='readSofa';ch.dir=0;}
      }
    }else{
      var spd=ch.speed*dt;ch.x+=(dx/dist)*Math.min(spd,dist);ch.y+=(dy/dist)*Math.min(spd,dist);
      ch.dir=dx>0?0:1;ch.pose='walk';ch.fTimer+=dt;
      if(ch.fTimer>0.2){ch.fTimer=0;ch.frame=1-ch.frame;}
    }
  }
  if(ch.state==='idle'&&!ch.moving){if(ch.sTimer>ch.idleWait){ch.sTimer=0;ch.idleWait=2+Math.random()*3;
    var t=randomIdleTarget();ch.path=buildPath(t.room,t.x,t.y);ch.pathIdx=0;ch.moving=true;}}
  if(ch.state==='excited'){ch.jumpT+=dt;ch.jumpOff=Math.round(Math.sin(ch.jumpT*8)*5);}
  var idle=Date.now()-ch.lastAct;
  if(ch.state==='idle'&&idle>60000)setState('reading');
  else if(ch.state==='reading'&&idle>180000)setState('sleeping');
  if(ch.state==='sleeping'&&!ch.moving)ch.bubble='z'.repeat((Math.floor(ch.sTimer*1.5)%3)+1);
  if(ch.state==='thinking')ch.bubble='.'.repeat((Math.floor(ch.sTimer*2)%3)+1);
  /* Cat */
  if(cat.state==='idle'){cat.waitT-=dt;if(cat.waitT<=0){
    if(Math.random()<0.3){cat.state='sleep';cat.sleepT=5+Math.random()*8;}
    else{cat.tx=WP+40+Math.random()*260;cat.ty=LF;cat.moving=true;cat.state='walk';}}}
  if(cat.state==='walk'&&cat.moving){
    var dx=cat.tx-cat.x,dy=cat.ty-cat.y,dist=Math.sqrt(dx*dx+dy*dy);
    if(dist<3){cat.x=cat.tx;cat.y=cat.ty;cat.moving=false;cat.state='idle';cat.waitT=3+Math.random()*4;}
    else{var spd=60*dt;cat.x+=(dx/dist)*Math.min(spd,dist);cat.y+=(dy/dist)*Math.min(spd,dist);
      cat.dir=dx>0?0:1;cat.fTimer+=dt;if(cat.fTimer>0.25){cat.fTimer=0;cat.frame=1-cat.frame;}}}
  if(cat.state==='sleep'){cat.sleepT-=dt;if(cat.sleepT<=0){cat.state='idle';cat.waitT=2+Math.random()*3;}}
}

/* ═══ RENDER ═══ */
function render(){
  if(bgOK)ctx.drawImage(bgC,0,0);
  else{ctx.fillStyle='#18161a';ctx.fillRect(0,0,W,H);}
  drawEnvAnimations();
  /* Draw cat scaled up */
  ctx.save();ctx.imageSmoothingEnabled=false;ctx.translate(cat.x,cat.y);ctx.scale(1.3,1.3);ctx.translate(-cat.x,-cat.y);
  drawCat(cat.x,cat.y,cat.dir,cat.state==='sleep');
  ctx.restore();
  drawChar();
  if(ch.bubble)drawBubble(ch.x,ch.y);
  ctx.fillStyle='rgba(255,245,230,0.015)';ctx.fillRect(0,0,W,H);
}

/* ═══ MAIN LOOP ═══ */
buildBackground();
var last=0;
function loop(t){var dt=Math.min((t-last)/1000,0.1);last=t;update(dt);render();requestAnimationFrame(loop);}
setState('idle');
requestAnimationFrame(function(t){last=t;requestAnimationFrame(loop);});
window.pixelRoomSetState=function(s){setState(s);};
}
_boot();
})();
"""
