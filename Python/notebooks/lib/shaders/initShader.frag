#version 300 es
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
 * initShader   :   Initialize Beeler-Reuter Variables 
 *
 * PROGRAMMER   :   ABOUZAR KABOUDIAN
 * DATE         :   Wed 19 Jul 2017 12:31:30 PM EDT
 * PLACE        :   Chaos Lab @ GaTech, Atlanta, GA
 *@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
 */
precision highp float;

/*------------------------------------------------------------------------
 * Interface variables : 
 * varyings change to "in" types in fragment shaders 
 * and "out" in vertexShaders
 *------------------------------------------------------------------------
 */
in vec2 pixPos ;

/*------------------------------------------------------------------------
 * It turns out for my current graphics card the maximum number of 
 * drawBuffers is limited to 8 
 *------------------------------------------------------------------------
 */
layout (location = 0 )  out vec4 outFvfs ;
layout (location = 1 )  out vec4 outSvfs ;


/*========================================================================
 * Main body of the shader
 *========================================================================
 */
void main() {
    vec4 vfs = vec4(0.0,1.0,0.4,0.0) ;

    if (pixPos.x<0.05){
        vfs.r = 1.0 ;
    }

    outFvfs = vfs ;
    outSvfs = vfs ;
    return ;
}
