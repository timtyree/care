#version 300 es
/*========================================================================
 * tipShader2D - adapted from tiptShader.frag from Abubu.js by Kaboudian (2019)
 *
 * PROGRAMMER   :   TIMOTHY TYREE
 * DATE         :   Wed 26 Jun 2019 01:28:07 PM EDT
 * PLACE        :   Rappel Lab @ UC San Diego, CA
 *========================================================================
 */
#include precision.glsl

/*========================================================================
 * Interface Variables
 *========================================================================
 */
in      vec2 pixPos ;
uniform sampler2D  vPrv;
uniform sampler2D  vCur;
uniform sampler2D  tips ;
uniform float Uth ;

uniform bool path ;

layout (location = 0 )  out vec4 ftipt ;

/*=========================================================================
 * main
 *=========================================================================
 */
void main(){
    if (!path){
        ftipt = vec4( 0., 0., 0., 0.) ;
        return ;
    }

    vec4 tip = vec4( 0., 0., 0., 0.) ; 
    vec2 size = vec2( textureSize(vPrv,0)) ;
    vec2 ii = vec2(1.,0.)/size  ;
    vec2 jj = vec2(0.,1.)/size  ;

    float v0 = texture( vCur, pixPos ).r ;
    float vx = texture( vCur, pixPos + ii).r ;
    float vy = texture( vCur, pixPos + jj).r ;
    float vxy= texture( vCur, pixPos + ii+ jj).r ;

    float f0 = v0 - Uth ;
    float fx = vx - Uth ;
    float fy = vy - Uth ;
    float fxy= vxy - Uth ;

    float s = step(0., f0) + step(0., fx) + step(0., fy) +
        step(0., fxy ) ;
    bool bv = (s>.5) && ( s<3.5 ) ;

    float d0    = v0   -   texture( vPrv, pixPos          ).r ; 
    float dx    = vx   -   texture( vPrv, pixPos + ii     ).r ;
    float dy    = vy   -   texture( vPrv, pixPos + jj     ).r ;
    float dxy   = vxy  -   texture( vPrv, pixPos + ii + jj).r ;

    s = step(0., d0) + step(0., dx) + step(0., dy) + step(0.,dxy) ;
    bool bdv = (s>.25) && ( s<3.5 ) ;

    if( !( (tip.r > .5) || (bv && bdv) ) ){
        ftipt = tip ;
        return;
    }

    fx -= f0;  fy -= f0;  dx -= d0;  dy -= d0;
    float det1 = 1./(fx*dy - fy*dx);
    float x = -(f0*dy - d0*fy)*det1; // intersection coordinates
    float y = -(fx*d0 - dx*f0)*det1;
    if  ( (x > 0.) && (x < 1.) && (y > 0.) && (y < 1.) ){
        tip = vec4(1.,1.,1.,1.) ;
    }

    ftipt = tip ;
    return ;
}
