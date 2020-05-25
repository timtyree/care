#version 300 es
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
 * pointShader   :   Minimal Atrial Model 
 *
 * PROGRAMMER   :   Timothy Tyree
 * DATE         :   Tue 2 Jul 2019 10:55 PM PST
 * PLACE        :   Rappel Lab @ UC, San Diego, CA
 *@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
 */
precision highp float;
precision highp int ;

/*------------------------------------------------------------------------
 * Interface variables : 
 * varyings change to "in" types in fragment shaders 
 * and "out" in vertexShaders
 *------------------------------------------------------------------------
 */
in vec2 pixPos ;

uniform sampler2D   inTrgt, tipts, state ;
uniform float       height, width ;
uniform float       Ifi_discrim   ;
#define thick       1
#define Uth         0.9             

/*------------------------------------------------------------------------
 * It turns out for my current graphics card the maximum number of 
 * drawBuffers is limited to 8 
 *------------------------------------------------------------------------
 */
layout (location = 0 ) out    vec4 outTrgt ;


/*========================================================================
 * Main body of the shader
 *========================================================================
 */
void main() {
    vec4    inVal   = texture( inTrgt, pixPos ) ;
    if(inVal.r < Uth){
        outTrgt = vec4(0.) ;
        return ;
    }

    //discriminate on the basis of I_fi
    //if (texture(state, pixPos).y > Ifi_discrim){
    //    outTrgt = vec4(0.) ;
    //    return ;
    //}
/*------------------------------------------------------------------------
 * output a tip if all nearest neighbors are zero
 *------------------------------------------------------------------------
 */
    vec2    cc      = pixPos ;
    vec2    size    = vec2(textureSize( inTrgt, 0 ) );
    //float   cddx    = size.x/ds_x ;
    //float   cddy    = size.y/ds_y ;

    vec2 ii = vec2(1.0,0.0)/size ;
    vec2 jj = vec2(0.0,1.0)/size ;
    vec2 ds = (ii + jj)/2. ;

    vec2 n  = cc + jj ;
    vec2 e  = cc + ii ;
    vec2 s  = cc - jj ;
    vec2 w  = cc - ii ;
    vec2 ne = cc + jj + ii ;
    vec2 nw = cc + jj - ii ;
    vec2 se = cc - jj + ii ;
    vec2 sw = cc - jj - ii ;

    bool v0 = texture( inTrgt, cc ).r>Uth ;
    bool vn = texture( inTrgt, n  ).r>Uth ;
    bool vs = texture( inTrgt, s  ).r>Uth ;
    bool ve = texture( inTrgt, e  ).r>Uth ;
    bool vw = texture( inTrgt, w  ).r>Uth ;
    bool vne= texture( inTrgt, ne ).r>Uth ;
    bool vnw= texture( inTrgt, nw ).r>Uth ;
    bool vse= texture( inTrgt, se ).r>Uth ;
    bool vsw= texture( inTrgt, sw ).r>Uth ;

    bool S = false ;
    S = S || vn ;
    S = S || vs ;
    S = S || ve ;
    S = S || vw ; 
    S = S || vne;
    S = S || vnw;
    S = S || vse;
    S = S || vsw; 

    if(!S){
      vec4 ions = texture(state, cc) ;
      //outTrgt = vec4(1., ions.yzw) ;
      //outTrgt = vec4(1.,1.,cc) ;
      outTrgt   = vec4(-ions.wy, cc ) ;
      return ;
    }

    float N  = 0. ;
    N += (vn)  ?  1.:0.;
    N += (vs)  ?  1.:0.;
    N += (ve)  ?  1.:0.;
    N += (vw)  ?  1.:0.;
    N += (vne) ?  1.:0.;
    N += (vnw) ?  1.:0.;
    N += (vse) ?  1.:0.;
    N += (vsw) ?  1.:0.;

    vec2 avg     =   cc        ;
    avg += (vn)  ?   n:vec2(0.);
    avg += (vs)  ?   s:vec2(0.);
    avg += (ve)  ?   e:vec2(0.);
    avg += (vw)  ?   w:vec2(0.);
    avg += (vne) ?  ne:vec2(0.);
    avg += (vnw) ?  nw:vec2(0.);
    avg += (vse) ?  se:vec2(0.);
    avg += (vsw) ?  sw:vec2(0.);
    avg /= N ;

    //if the average position is within a half voxel of averagePos(), then return true
    S = true ; 
    S = S && ( avg.x-ds.x < cc.x && cc.x <= avg.x+ds.x ) ;
    S = S && ( avg.y-ds.y < cc.y && cc.y <= avg.y+ds.y ) ;

    if(S){
    vec4 ions = texture(state, cc) ;
    //outTrgt = vec4(1., ions.yzw) ;// state  is vec4(vlt,Ifi, Iso, Isi);
    //outTrgt = vec4(1.,1.,avg) ;
    outTrgt   = vec4(-ions.wy, avg ) ;
    return ;
    }

    outTrgt = vec4(0.) ;
    return ;       
}