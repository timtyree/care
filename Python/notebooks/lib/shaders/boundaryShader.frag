#version 300 es
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
 * compShader   :   Minimal Atrial Model 
 *
 * PROGRAMMER   :   ABOUZAR KABOUDIAN, Timothy Tyree
 * DATE         :   Tue 12 Jun 2018 02:22:27 PM EDT
 * PLACE        :   Chaos Lab @ GaTech, Atlanta, GA
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

uniform sampler2D   inTrgt ;
uniform vec3        domainResolution ;
uniform vec3        domainSize ;

uniform sampler2D   phaseTxt , nsewAvgTxt , updnAvgTxt ;
uniform sampler2D   nhshMapTxt, etwtMapTxt, updnMapTxt ;

/*------------------------------------------------------------------------
 * It turns out for my current graphics card the maximum number of 
 * drawBuffers is limited to 8 
 *------------------------------------------------------------------------
 */
layout (location = 0 )  out vec4 outTrgt ;

/*========================================================================
 * Main body of the shader
 *========================================================================
 */
void main() {
/*------------------------------------------------------------------------
 * reading from textures
 *------------------------------------------------------------------------
 */
    vec4    phasVal = texture( phaseTxt, pixPos ) ;
    vec4    inVal   = texture( inTrgt, pixPos ) ;

/*------------------------------------------------------------------------
 * check if we are outside domain
 *------------------------------------------------------------------------
 */
    if (phasVal.r < 0.01){
        outTrgt = inVal ;
        return ;
    }

/*-------------------------------------------------------------------------
 * Extracting spatially local local variables
 *-------------------------------------------------------------------------
 */
    vec4 nhshMap = texture(nhshMapTxt, pixPos ) ;
    vec4 etwtMap = texture(etwtMapTxt, pixPos ) ;
    vec4 updnMap = texture(updnMapTxt, pixPos ) ;

    float   c   = texture(inTrgt, pixPos     ).r ; //c is the red channel at the central pixel
    float   n   = texture(inTrgt, nhshMap.xy ).g ;
    float   s   = texture(inTrgt, nhshMap.zw ).g ;
    float   e   = texture(inTrgt, etwtMap.xy ).g ;
    float   w   = texture(inTrgt, etwtMap.zw ).g ;//w is the green channel at the pixel neighboring to the west
    float   u   = texture(inTrgt, updnMap.xy ).g ;//u is the green channel at the pixel neighboring up above the current pixel.
    float   d   = texture(inTrgt, updnMap.zw ).g ;

    /*   TT: As of writing this initial paragraph, the [activation] front is put in the 'r' channel and the [deactivation] front is put in the 'g' channel.  We want to highlight the set of points that either (i) are 'r' and neighbor (ii) a 'g', or are 'g' and neighbor a 'r'.  Wlog, we choose only (i) to get a thinner boundary. Nevermind, I think you need to include (i) and (ii) to get reliable spiral tips. 

    Since including (i) and (ii) prevents denoising, tracking tips becomes substantially more difficult since the output 'r' channel now carries noise.  We hope to address this noise with a denoising convolutional autoencoder.  If this does not work, we should consider applying feature detection on the input data and then boolean thresholding on the output.  If either of those last two methods proves cumbersome, we would consider using machine learning constrained to the appropriate symmetries.  */
    
    float   c2   = texture(inTrgt, pixPos     ).g ; //c is the red channel at the central pixel
    float   n2   = texture(inTrgt, nhshMap.xy ).r ;
    float   s2   = texture(inTrgt, nhshMap.zw ).r ;
    float   e2   = texture(inTrgt, etwtMap.xy ).r ;
    float   w2   = texture(inTrgt, etwtMap.zw ).r ;//w is the green channel at the pixel neighboring to the west
    float   u2   = texture(inTrgt, updnMap.xy ).r ;//u is the green channel at the pixel neighboring up above the current pixel.
    float   d2   = texture(inTrgt, updnMap.zw ).r ;


/*-------------------------------------------------------------------------
* Boolean computation of boundary between fronts
*-------------------------------------------------------------------------
*/
    float thresh = 0.5;
    float p = 0.0;
    
    // condition (i)
    if (c >thresh) {
        //check for neighbors that are green
        p = (n >thresh || s >thresh || e >thresh || w >thresh || u >thresh || d >thresh) ? 1.0:p;
    }
    
    // condition (ii)
    if (c2 >thresh) {
        //check for neighbors that are green
        p = (n2 >thresh || s2 >thresh || e2 >thresh || w2 >thresh || u2 >thresh || d2 >thresh) ? 1.0:p;
    }


/*------------------------------------------------------------------------
 * ouputing the shader
 *------------------------------------------------------------------------
 */

    outTrgt = vec4(p,0.,0.,0.);
    return ;
}
