#version 300 es
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
 * compShader   :   Beeler-Reuter Compute Shader with periodic b.c. outputing current states
 *
 * PROGRAMMER   :   Timothy Tyree
 * Adapted From :   ABOUZAR KABOUDIAN
 * DATE         :   Wed 3 Jul 2019 11:17 PM PST
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

uniform sampler2D   inVfs ;

uniform float       ds_x, ds_y ;
uniform float       dt ;
uniform float       diffCoef, C_m ;

uniform float       tau_pv, 
                    tau_v1, 
                    tau_v2, 
                    tau_pw, 
                    tau_mw, 
                    tau_d, 
                    tau_0, 
                    tau_r,
                    tau_si,
                    K,
                    V_sic, 
                    V_c,
                    V_v ;

uniform float   C_si ;

#define vSampler    inVfs
#define Uth         0.9
#define size        vec2(textureSize( vSampler, 0 ))
#define dx          1./size.x
#define dy          1./size.y 

/*------------------------------------------------------------------------
 * It turns out for my current graphics card the maximum number of 
 * drawBuffers is limited to 8 
 *------------------------------------------------------------------------
 */
layout (location = 0 )  out vec4 outVfs ;
layout (location = 1 )  out vec4 state  ;

/*========================================================================
 * Tanh
 *========================================================================
 */
float Tanh(float x){
    if ( x < -3.) return -1. ;
    if ( x > 3. ) return 1.  ;
    else return x*(27.+x*x)/(27.+9.*x*x) ;
}

/*------------------------------------------------------------------------
 * applying periodic boundary conditions for each texture call
 *------------------------------------------------------------------------
 */ 

vec4 mytexture(sampler2D S, vec2 cc){
    if ( cc.x < dx  ){ // Left P.B.C.
        cc.x = 1.   ;
    }else if ( cc.x > (1. - dx) ){ // Right P.B.C.
        cc.x = 0. ;
    }else if( cc.y <  dy ){    //  Bottom P.B.C.
        cc.y = 1. ;
    }else if ( cc.y > (1. - dy)){ // Top P.B.C.
        cc.y = 0.;
    }
    return texture(S, cc) ;
}

/*========================================================================
 * Main body of the shader
 *========================================================================
 */
void main() {
    vec2    cc = pixPos ;
    float   cddx    = size.x/ds_x ;
    float   cddy    = size.y/ds_y ;

    cddx *= cddx ;
    cddy *= cddy ;


/*------------------------------------------------------------------------
 * reading from textures
 *------------------------------------------------------------------------
 */
    vec4    C = mytexture( inVfs , pixPos ) ;
    float   vlt = C.r ;
    float   fig = C.g ;
    float   sig = C.b ;

/*-------------------------------------------------------------------------
 * Calculating right hand side vars
 *-------------------------------------------------------------------------
 */
    float p = step(V_c, vlt) ;
    float q = step(V_v, vlt) ;

    float tau_mv = (1.0-q)*tau_v1   +  q*tau_v2 ;

    float Ifi  = -fig*p*(vlt - V_c)*(1.0-vlt)/tau_d ;
    float Iso  =  vlt*(1.0  - p )/tau_0 + p/tau_r ;

    float tn = Tanh(K*(vlt-V_sic)) ;
    float Isi  = -sig*(1.0  + tn) /(2.0*tau_si) ;
    Isi *= C_si ;
    float dFig2dt  = (1.0-p)*(1.0-fig)/tau_mv - p*fig/tau_pv ;
    float dSig2dt  = (1.0-p)*(1.0-sig)/tau_mw - p*sig/tau_pw ;

    fig += dFig2dt*dt ;
    sig += dSig2dt*dt ;

/*-------------------------------------------------------------------------
 * Laplacian
 *-------------------------------------------------------------------------
 */
    vec2 ii = vec2(1.0,0.0)/size ;
    vec2 jj = vec2(0.0,1.0)/size ;    
    
    float gamma = 1./3. ;

    float dVlt2dt = (1.-gamma)*((   mytexture(vSampler,cc+ii).r
                                -   2.0*C.r
                                +   mytexture(vSampler,cc-ii).r     )*cddx
                            +   (   mytexture(vSampler,cc+jj).r
                                -   2.0*C.r
                                +   mytexture(vSampler,cc-jj).r     )*cddy  )

                +   gamma*0.5*(     mytexture(vSampler,cc+ii+jj).r
                                +   mytexture(vSampler,cc+ii-jj).r
                                +   mytexture(vSampler,cc-ii-jj).r
                                +   mytexture(vSampler,cc-ii+jj).r
                                -   4.0*C.r               )*(cddx + cddy) ;
    dVlt2dt *= diffCoef ;

/*------------------------------------------------------------------------
 * I_sum
 *------------------------------------------------------------------------
 */
    float I_sum = Isi + Ifi + Iso ;

/*------------------------------------------------------------------------
 * Time integration for membrane potential
 *------------------------------------------------------------------------
 */
    dVlt2dt -= I_sum/C_m ;
    vlt += dVlt2dt*dt ;

/*------------------------------------------------------------------------
 * ouputing the shader
 *------------------------------------------------------------------------
 */
    state  = vec4(vlt,Ifi, Iso, Isi);
    outVfs = vec4(vlt,fig,sig,1.0);
    return ;
}
