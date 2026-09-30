VSIN(ATTRIB_POS)	vec3 in_pos;

VSOUT vec4 v_color;
VSOUT vec2 v_tex0;
VSOUT float v_fog;

// Depth pull for world-space sprites (visionOS coronas): scale the VIEW-space position
// toward the eye by (1 + u_im3dPull.x). A homothety about the eye keeps the projected
// screen position of every vertex, only the depth changes -- exactly what the 2D sprite
// path did with z -= nearDist and the sun at 0.95*far, but per eye, so stereo disparity stays
// that of the true position. 0 (GL default, never set on other platforms) = no change.
uniform vec4 u_im3dPull;

void
main(void)
{
	vec4 Vertex = u_world * vec4(in_pos, 1.0);
	vec4 CamVertex = u_view * Vertex;
	CamVertex.xyz *= (1.0 + u_im3dPull.x);
	gl_Position = u_proj * CamVertex;
	v_color = in_color;
	v_tex0 = in_tex0;
	v_fog = DoFogV(CamVertex, gl_Position.w);
}
