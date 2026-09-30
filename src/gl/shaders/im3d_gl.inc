const char *im3d_vert_src =
"VSIN(ATTRIB_POS)	vec3 in_pos;\n"

"VSOUT vec4 v_color;\n"
"VSOUT vec2 v_tex0;\n"
"VSOUT float v_fog;\n"

"// Depth pull for world-space sprites (visionOS coronas): scale the VIEW-space position\n"
"// toward the eye by (1 + u_im3dPull.x). A homothety about the eye keeps the projected\n"
"// screen position of every vertex, only the depth changes -- exactly what the 2D sprite\n"
"// path did with z -= nearDist and the sun at 0.95*far, but per eye, so stereo disparity stays\n"
"// that of the true position. 0 (GL default, never set on other platforms) = no change.\n"
"uniform vec4 u_im3dPull;\n"

"void\n"
"main(void)\n"
"{\n"
"	vec4 Vertex = u_world * vec4(in_pos, 1.0);\n"
"	vec4 CamVertex = u_view * Vertex;\n"
"	CamVertex.xyz *= (1.0 + u_im3dPull.x);\n"
"	gl_Position = u_proj * CamVertex;\n"
"	v_color = in_color;\n"
"	v_tex0 = in_tex0;\n"
"	v_fog = DoFogV(CamVertex, gl_Position.w);\n"
"}\n"
;
